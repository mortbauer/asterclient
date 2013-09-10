from bs4 import BeautifulSoup
import requests
import sys
def utf8_encode_dict(data):
    utf8_encode = lambda x: x.encode('utf-8')
    return dict(map(utf8_encode, pair) for pair in data.items())

def translator_translate(text,sl,tl):
    params = {'sl':sl,'tl':tl,'text':text}
    headers = {'User-Agent':'Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) Gecko/20071127 Firefox/2.0.0.11'}
    request = requests.get('http://www.translate.google.com',params=params,headers=headers)
    feeddata = request.text
    soup = BeautifulSoup(feeddata)
    try:
        result = soup.find('span', id="result_box")
        return '\n'.join([x.get('title') for x in result.contents])
    except:
        return('No result found')

