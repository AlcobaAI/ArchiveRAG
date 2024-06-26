from base_module import Scraper
from scrape_utils import get_soup
import requests
import os

class IP1Scraper(Scraper):
    def __init__(self, config):
        super().__init__(config)

    def get_data(self, url):
        return super().get_data(url)

    def has_href(self, a_tag):
        return super().has_href(a_tag)

    def avoids_strings(self, a_tag):
        return super().avoids_strings(a_tag)

    def has_any_filter(self, a_tag):
        return super().has_any_filter(a_tag)

    def save_progress(self, urls, seen):
        super().save_progress(urls, seen)

    def download_pdf(self, url, save_path):
        """
        Downloads a PDF file from a given URL and saves it to a specified path.
    
        Args:
        url (str): The URL of the PDF file.
        save_path (str): The full path to where the file will be saved, including the filename.
    
        Returns:
        bool: True if the download was successful, False otherwise.
        """
        try:
            # Send a GET request to the URL
            response = requests.get(url)
            
            # Check if the request was successful
            if response.status_code == 200:
                # Ensure the directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                # Write the response content to a file
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                print(f"File has been downloaded and saved to: {save_path}")
                return True
            else:
                print(f"Failed to download file: {url}")
                return False
        except Exception as e:
            print(f"An error occurred: {url}")
            return False

    def get_data(self, url):
        
        common_url = self.config['common_url']

        soup = get_soup(url)
        
        data = dict()
        
        try:
            text_elements = self.search_elements(soup)
            text_elements = [t for t in text_elements if t.text.strip() != '']
            text = {f"{n} - {p.name}":p.text for n, p in enumerate(text_elements)}
            #text = '\n'.join([p.text for p in soup.find(content_tag, class_ = content_class).findAll({'p', 'h1', 'h2', 'h3', 'h4', 'li'})])
            data['url'] = url
            data['text'] = text
        except Exception as e:
            print(e)
            pass

        pdf_links = [a['href'].replace('../', '') for a in soup.findAll('a') if self.has_href(a) and '.pdf' in a['href']]
        pdf_links = [p if p.startswith('http') else 'http://www.interpares.org/' + p for p in pdf_links]

        for link in pdf_links:
            filename = link.split('/')[-1].split('doc=')[-1]
            os.makedirs('ip1', exist_ok=True)
            if filename in os.listdir('ip1'):
                continue
            
            save_path = f'ip1/{filename}'
            self.download_pdf(link, save_path)
    
        new_urls = [a for a in soup.findAll('a') if self.has_href(a) and self.has_any_filter(a) and self.avoids_strings(a)]
        new_urls = ['ip1/' + a['href'] if 'http' not in a['href'] and 'ip1/' not in a['href'] else a['href'] for a in new_urls]
        new_urls = [common_url + a if 'http' not in a else a for a in new_urls]
        new_urls = [u for u in new_urls if common_url in u]

        data['new_urls'] = new_urls
        
        return data

    def search_elements(self, soup):
        return super().search_elements(soup)
        
    def parse_urls(self):
        super().parse_urls()
    
    def process_data(self, data):
        super().process_data(data)