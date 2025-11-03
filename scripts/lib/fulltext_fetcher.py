#!/usr/bin/env python3
"""
Full-text fetcher for academic papers.

Attempts to retrieve full text from multiple sources:
1. Unpaywall API (open access)
2. Europe PMC (PubMed Central)
3. Optional institutional proxy

Handles PDF extraction, text normalization, and caching.
"""
import re
import time
import requests
from pathlib import Path
from typing import Optional, Dict
from urllib.parse import quote
import diskcache
import io

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False


class FullTextFetcher:
    """
    Fetch full text from academic papers using multiple sources.

    Features:
    - Multi-source fallback (Unpaywall → Europe PMC → Proxy)
    - PDF text extraction
    - Text normalization for matching
    - Disk caching (24 hour TTL)
    - Rate limiting
    """

    def __init__(self,
                 email: str,
                 cache_dir: str = '.cache/fulltext',
                 proxy_url: Optional[str] = None,
                 proxy_cookies: Optional[Dict] = None,
                 rate_limit: float = 1.0):
        """
        Initialize full-text fetcher.

        Args:
            email: Email for Unpaywall API (required)
            cache_dir: Directory for caching full text
            proxy_url: Optional institutional proxy URL pattern
            proxy_cookies: Optional auth cookies for proxy
            rate_limit: Seconds between requests (default: 1.0)
        """
        self.email = email
        self.proxy_url = proxy_url
        self.proxy_cookies = proxy_cookies
        self.rate_limit = rate_limit
        self.last_request_time = 0

        # Initialize cache
        self.cache = diskcache.Cache(cache_dir)
        self.cache_ttl = 86400  # 24 hours

        if not PDFPLUMBER_AVAILABLE:
            print("Warning: pdfplumber not available. PDF extraction will fail.")

    def _rate_limit_wait(self):
        """Enforce rate limiting between requests."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()

    def get_full_text(self, doi: str) -> Dict:
        """
        Get full text for a DOI from any available source.

        Args:
            doi: DOI to fetch

        Returns:
            Dict with keys:
            - 'text': Full text string or None
            - 'source': Source name if successful ('unpaywall', 'europepmc', 'proxy') or None
            - 'attempts': List of dicts describing each attempt
        """
        result = {
            'text': None,
            'source': None,
            'attempts': []
        }

        if not doi:
            result['attempts'].append({
                'source': 'validation',
                'status': 'invalid_doi',
                'message': 'Empty DOI'
            })
            return result

        # Normalize DOI
        doi = doi.strip().lower()
        if doi.startswith('http'):
            # Extract DOI from URL
            match = re.search(r'10\.\d+/[^\s]+', doi)
            if match:
                doi = match.group(0)
            else:
                result['attempts'].append({
                    'source': 'validation',
                    'status': 'invalid_doi',
                    'message': f'Could not extract DOI from URL: {doi[:50]}'
                })
                return result

        # Check cache
        cache_key = f"fulltext:{doi}"
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            # Return cached result (which is already a dict)
            if isinstance(cached_result, dict):
                # Only return if it's a successful fetch (has text)
                # Ignore cached failures from old code - retry them
                if cached_result.get('text'):
                    return cached_result
                # Fall through to retry if cached failure
            else:
                # Legacy cache format (just text string) - convert and return
                result['text'] = cached_result
                result['source'] = 'cache'
                return result

        # Try sources in order
        # 1. Try Unpaywall (open access)
        text, attempt_info = self._try_unpaywall(doi)
        result['attempts'].append(attempt_info)
        if text:
            result['text'] = text
            result['source'] = 'unpaywall'
            # Only cache successful fetches (24 hour TTL)
            self.cache.set(cache_key, result, expire=self.cache_ttl)
            return result

        # 2. Try Europe PMC (PubMed Central)
        text, attempt_info = self._try_europepmc(doi)
        result['attempts'].append(attempt_info)
        if text:
            result['text'] = text
            result['source'] = 'europepmc'
            # Only cache successful fetches (24 hour TTL)
            self.cache.set(cache_key, result, expire=self.cache_ttl)
            return result

        # 3. Try institutional proxy (if configured)
        if self.proxy_url and self.proxy_cookies:
            text, attempt_info = self._try_proxy(doi)
            result['attempts'].append(attempt_info)
            if text:
                result['text'] = text
                result['source'] = 'proxy'
                # Only cache successful fetches (24 hour TTL)
                self.cache.set(cache_key, result, expire=self.cache_ttl)
                return result
        else:
            result['attempts'].append({
                'source': 'proxy',
                'status': 'not_configured',
                'message': 'Institutional proxy not configured'
            })

        # DO NOT cache failures - always retry on next run
        # This allows retries when:
        # - Papers become open access
        # - Proxy gets configured
        # - Temporary API errors resolve
        return result

    def _try_unpaywall(self, doi: str) -> tuple:
        """
        Try fetching from Unpaywall API.

        Args:
            doi: DOI to fetch

        Returns:
            Tuple of (text, attempt_info)
        """
        attempt_info = {'source': 'unpaywall', 'status': 'unknown', 'message': ''}

        try:
            self._rate_limit_wait()

            url = f"https://api.unpaywall.org/v2/{quote(doi)}?email={self.email}"
            response = requests.get(url, timeout=10)

            if response.status_code != 200:
                attempt_info['status'] = 'api_error'
                attempt_info['message'] = f"HTTP {response.status_code}"
                attempt_info['raw_response'] = {
                    'status_code': response.status_code,
                    'text': response.text[:500] if response.text else None
                }
                return (None, attempt_info)

            data = response.json()

            # Store raw API response for debugging
            attempt_info['raw_response'] = data

            # Check if OA available
            if not data.get('is_oa'):
                attempt_info['status'] = 'not_open_access'
                attempt_info['message'] = 'Paper is not open access'
                return (None, attempt_info)

            # Get best OA location
            best_oa = data.get('best_oa_location')
            if not best_oa:
                attempt_info['status'] = 'no_oa_location'
                attempt_info['message'] = 'No OA location found'
                return (None, attempt_info)

            pdf_url = best_oa.get('url_for_pdf')
            if pdf_url:
                attempt_info['pdf_url'] = pdf_url
                text = self._fetch_pdf(pdf_url)
                if text:
                    attempt_info['status'] = 'success'
                    attempt_info['message'] = f'Retrieved PDF from {pdf_url[:50]}...'
                    return (text, attempt_info)
                else:
                    attempt_info['status'] = 'extraction_failed'
                    attempt_info['message'] = f'PDF found at {pdf_url} but extraction failed'
                    return (None, attempt_info)

            # Try landing page (might have HTML full text)
            landing_url = best_oa.get('url')
            if landing_url:
                attempt_info['landing_url'] = landing_url
                text = self._fetch_html(landing_url)
                if text:
                    attempt_info['status'] = 'success'
                    attempt_info['message'] = f'Retrieved HTML from {landing_url[:50]}...'
                    return (text, attempt_info)
                else:
                    attempt_info['status'] = 'extraction_failed'
                    attempt_info['message'] = f'HTML found at {landing_url} but extraction failed'
                    return (None, attempt_info)

            attempt_info['status'] = 'extraction_failed'
            attempt_info['message'] = 'Found OA location but could not extract text'
            attempt_info['raw_response'] = {'best_oa_location': best_oa}
            return (None, attempt_info)

        except Exception as e:
            attempt_info['status'] = 'error'
            attempt_info['message'] = f"Exception: {str(e)}"
            return (None, attempt_info)

    def _try_europepmc(self, doi: str) -> tuple:
        """
        Try fetching from Europe PMC.

        Args:
            doi: DOI to fetch

        Returns:
            Tuple of (text, attempt_info)
        """
        attempt_info = {'source': 'europepmc', 'status': 'unknown', 'message': ''}

        try:
            self._rate_limit_wait()

            # Search for article
            search_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
            params = {
                'query': f'DOI:"{doi}"',
                'format': 'json',
                'resultType': 'core'
            }

            response = requests.get(search_url, params=params, timeout=10)
            if response.status_code != 200:
                attempt_info['status'] = 'api_error'
                attempt_info['message'] = f"Search API HTTP {response.status_code}"
                attempt_info['raw_response'] = {
                    'status_code': response.status_code,
                    'text': response.text[:500] if response.text else None
                }
                return (None, attempt_info)

            data = response.json()

            # Store raw API response for debugging
            attempt_info['raw_response'] = data

            results = data.get('resultList', {}).get('result', [])

            if not results:
                attempt_info['status'] = 'not_found'
                attempt_info['message'] = 'DOI not found in Europe PMC'
                return (None, attempt_info)

            # Get first result
            article = results[0]

            # Try to get full text URLs from fullTextUrlList first
            # (Check this regardless of isOpenAccess flag, as Europe PMC sometimes
            # has free URLs even when isOpenAccess != 'Y')
            full_text_url_list = article.get('fullTextUrlList', {}).get('fullTextUrl', [])

            # Check if any free URLs are available
            has_free_urls = any(
                url.get('availabilityCode') == 'F'
                for url in full_text_url_list
            )

            # If no free URLs and not open access, bail out early
            if not has_free_urls and article.get('isOpenAccess') != 'Y':
                attempt_info['status'] = 'not_open_access'
                attempt_info['message'] = 'Article found but not open access'
                return (None, attempt_info)

            # Skip HTML from Europe PMC (europepmc.org/articles/* are JS-rendered)
            # Go straight to PDF or XML which have raw content

            # Try free PDFs first
            for url_entry in full_text_url_list:
                if (url_entry.get('availabilityCode') == 'F' and
                    url_entry.get('documentStyle') == 'pdf'):
                    pdf_url = url_entry.get('url')
                    if pdf_url:
                        attempt_info['pdf_url'] = pdf_url
                        text = self._fetch_pdf(pdf_url)
                        if text:
                            attempt_info['status'] = 'success'
                            attempt_info['message'] = f'Retrieved PDF from Europe PMC: {pdf_url[:50]}...'
                            return (text, attempt_info)
                        else:
                            attempt_info['status'] = 'extraction_failed'
                            attempt_info['message'] = f'PDF found at {pdf_url} but extraction failed'
                            # Don't return yet, try XML as final fallback

            # Fall back to PMCID-based XML approach
            pmcid = article.get('pmcid')
            if not pmcid:
                attempt_info['status'] = 'no_pmcid'
                attempt_info['message'] = 'No PMCID available and no free PDF/HTML URLs worked'
                return (None, attempt_info)

            # Fetch full text XML
            fulltext_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
            response = requests.get(fulltext_url, timeout=10)

            if response.status_code != 200:
                attempt_info['status'] = 'fulltext_error'
                attempt_info['message'] = f"Full text API HTTP {response.status_code}"
                attempt_info['raw_response_fulltext'] = {
                    'status_code': response.status_code,
                    'text': response.text[:500] if response.text else None
                }
                return (None, attempt_info)

            # Extract text from XML (simple approach)
            xml_text = response.text

            # Store raw XML response info (first 1000 chars for debugging)
            attempt_info['raw_response_fulltext'] = {
                'status_code': response.status_code,
                'xml_preview': xml_text[:1000] if xml_text else None,
                'full_length': len(xml_text) if xml_text else 0
            }

            # Remove XML tags
            text = re.sub(r'<[^>]+>', ' ', xml_text)
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text).strip()

            if text:
                attempt_info['status'] = 'success'
                attempt_info['message'] = f'Retrieved XML for PMCID {pmcid}'
                return (text, attempt_info)

            attempt_info['status'] = 'extraction_failed'
            attempt_info['message'] = 'Full text XML was empty'
            return (None, attempt_info)

        except Exception as e:
            attempt_info['status'] = 'error'
            attempt_info['message'] = f"Exception: {str(e)}"
            return (None, attempt_info)

    def _try_proxy(self, doi: str) -> tuple:
        """
        Try fetching through institutional proxy.

        Args:
            doi: DOI to fetch

        Returns:
            Tuple of (text, attempt_info)
        """
        attempt_info = {'source': 'proxy', 'status': 'unknown', 'message': ''}

        try:
            self._rate_limit_wait()

            # Construct proxied URL
            target_url = f"https://doi.org/{quote(doi)}"
            proxy_url = f"{self.proxy_url}{target_url}"

            # Attempt to fetch with cookies
            response = requests.get(
                proxy_url,
                cookies=self.proxy_cookies,
                timeout=10,
                allow_redirects=True
            )

            # Store raw response info for debugging
            content_type = response.headers.get('content-type', '').lower()
            attempt_info['raw_response'] = {
                'status_code': response.status_code,
                'content_type': content_type,
                'content_length': len(response.content) if hasattr(response, 'content') else None,
                'url': proxy_url
            }

            if response.status_code != 200:
                attempt_info['status'] = 'http_error'
                attempt_info['message'] = f"HTTP {response.status_code}"
                return (None, attempt_info)

            # Check if we got PDF
            if 'pdf' in content_type:
                text = self._extract_pdf_text(response.content)
                if text:
                    attempt_info['status'] = 'success'
                    attempt_info['message'] = f'Retrieved PDF via proxy'
                    return (text, attempt_info)
                else:
                    attempt_info['status'] = 'extraction_failed'
                    attempt_info['message'] = 'PDF downloaded but text extraction failed'
                    return (None, attempt_info)

            # Try HTML extraction
            text = self._extract_html_text(response.text)
            if text:
                attempt_info['status'] = 'success'
                attempt_info['message'] = f'Retrieved HTML via proxy'
                return (text, attempt_info)

            attempt_info['status'] = 'extraction_failed'
            attempt_info['message'] = 'Response received but text extraction failed'
            return (None, attempt_info)

        except Exception as e:
            attempt_info['status'] = 'error'
            attempt_info['message'] = f"Exception: {str(e)}"
            return (None, attempt_info)

    def _fetch_pdf(self, url: str) -> Optional[str]:
        """Fetch and extract text from PDF URL."""
        try:
            self._rate_limit_wait()
            response = requests.get(url, timeout=30)

            if response.status_code != 200:
                return None

            return self._extract_pdf_text(response.content)

        except Exception as e:
            print(f"PDF fetch error: {e}")
            return None

    def _fetch_html(self, url: str) -> Optional[str]:
        """Fetch and extract text from HTML URL."""
        try:
            self._rate_limit_wait()
            response = requests.get(url, timeout=10)

            if response.status_code != 200:
                return None

            return self._extract_html_text(response.text)

        except Exception as e:
            print(f"HTML fetch error: {e}")
            return None

    def _extract_pdf_text(self, pdf_content: bytes) -> Optional[str]:
        """Extract text from PDF bytes."""
        if not PDFPLUMBER_AVAILABLE:
            print("Warning: pdfplumber not available")
            return None

        try:
            with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                text_parts = []
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)

                full_text = '\n'.join(text_parts)
                return self.normalize_text(full_text) if full_text else None

        except Exception as e:
            print(f"PDF extraction error: {e}")
            return None

    def _extract_html_text(self, html: str) -> Optional[str]:
        """Extract text from HTML (basic approach)."""
        try:
            # Remove script and style tags
            text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)

            # Remove HTML tags
            text = re.sub(r'<[^>]+>', ' ', text)

            # Decode HTML entities
            text = text.replace('&nbsp;', ' ')
            text = text.replace('&amp;', '&')
            text = text.replace('&lt;', '<')
            text = text.replace('&gt;', '>')
            text = text.replace('&quot;', '"')

            return self.normalize_text(text) if text else None

        except Exception as e:
            print(f"HTML extraction error: {e}")
            return None

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize text for matching.

        Handles:
        - Hyphenation at line breaks
        - Extra whitespace
        - Common PDF artifacts

        Args:
            text: Raw text

        Returns:
            Normalized text
        """
        if not text:
            return ""

        # Remove hyphenation at line breaks (e.g., "param-\neter" → "parameter")
        text = re.sub(r'-\s*\n\s*', '', text)

        # Replace line breaks with spaces
        text = text.replace('\n', ' ')

        # Normalize multiple spaces to single space
        text = re.sub(r'\s+', ' ', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def clear_cache(self):
        """Clear the full-text cache."""
        self.cache.clear()

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            'size': len(self.cache),
            'directory': self.cache.directory
        }
