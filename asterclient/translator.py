from bs4 import BeautifulSoup
import requests

class Translator(object):
    HEADERS = {'User-Agent':'Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) Gecko/20071127 Firefox/2.0.0.11'}
    BASE_URL = 'http://www.translate.google.com'
    CHUNK_SIZE = 1000

    def __init__(self,text,sl,tl):
        self.sl = sl
        self.tl =tl
        self.text = text

    def _get_params(self,text):
        return {'sl':self.sl,'tl':self.tl,'text':text}

    def _ask_google(self,text):
        request = requests.get(
            self.BASE_URL,params=self._get_params(text),headers=self.HEADERS)
        return request.text

    def _get_result(self,html):
        soup = BeautifulSoup(html)
        result = soup.find('span', id="result_box")
        return [x.text for x in result.contents]

    def get(self):
        self.result = []
        for i in range(int(len(self.text)/self.CHUNK_SIZE)):
            self.result.extend(
                self._get_result(
                    self._ask_google(
                        self.text[i*self.CHUNK_SIZE:(i+1)*self.CHUNK_SIZE])))
        self.result.extend(
            self._get_result(
                self._ask_google(
                        self.text[(i+1)*self.CHUNK_SIZE:])))
        return '\n'.join(self.result)



