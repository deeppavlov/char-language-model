import scrapy

import re
import os


def create_path(path, file_name_is_in_path=False):
    if file_name_is_in_path:
        folder_list = path.split('/')[:-1]
    else:
        folder_list = path.split('/')
    if len(folder_list) > 0:
        if folder_list[0] == '':
            current_folder = '/'
        else:
            current_folder = folder_list[0]
        for idx, folder in enumerate(folder_list):
            if idx > 0:
                current_folder += ('/' + folder)
            if not os.path.exists(current_folder):
                os.mkdir(current_folder)


def split_to_path_and_name(path):
    parts = path.split('/')
    name = parts[-1]
    path = '/'.join(parts[:-1])
    return path, name


def loop_through_indices(filename, start_index):
    path, name = split_to_path_and_name(filename)
    if '.' in name:
        inter_list = name.split('.')
        extension = inter_list[-1]
        base = '.'.join(inter_list[:-1])
        base += '#%s'
        name = '.'.join([base, extension])

    else:
        name += '#%s'
    if path != '':
        base_path = '/'.join([path, name])
    else:
        base_path = name
    index = start_index
    while os.path.exists(base_path % index):
        index += 1
    return base_path % index


def add_index_to_filename_if_needed(filename, index=None):
    if index is not None:
        return loop_through_indices(filename, index)
    if os.path.exists(filename):
        return loop_through_indices(filename, 1)
    return filename


def filter_text(text):
    text = re.sub('\\xa0', ' ', text)
    text = re.sub('&nbsp;', ' ', text)
    text = re.sub(' '{2,}, ' ', text)
    return text


def get_elem_type(sel):
    expr = sel._expr
    last_elem = expr.split('/')[-1]
    if last_elem == 'text()':
        last_elem = 'text'
    return last_elem


def process_simple_text(sel):
    return ' ' + filter_text(''.join(sel.extract())) + ' '


def process_nobr(sel):
    return ' ' + filter_text(''.join(sel.xpath('text()').extract())) + ' '


def process_bold(bold_sel):
    return '<b>' + filter_text(''.join(bold_sel.xpath('text()').extract())) + '</b>'


def process_italic(italic_sel):
    return '<i>' + filter_text(''.join(italic_sel.xpath('text()').extract())) + '</i>'


def process_quote(quote_sel):
    text = ''
    sels = quote_sel.xpath('text() | i | b | nobr | blockquote')
    for sel in sels:
        text += process(sel)
    text = re.sub(' '{2,}, ' ', text)
    return text


def process(sel):
    mode = get_elem_type(sel)
    text = '' 
    if mode == 'elem_type':
        if mode == 'text':
            text += process_simple_text(sel)
        elif mode == 'b':
            text += process_bold(sel)
        elif mode == 'i':
            text += process_italic(sel)
        elif mode == 'blockquote':
            text += process_quote(sel)
        elif mode == 'nobr':
            text += process
    return text


def extract_text_and_additional_information(text_content):
    introduction = ''
    participants = list()
    text = ''
    bibliography = ''
    release_information = ''

    introduction_is_processed = True
    participants_are_processed = False
    text_is_processed = False
    bibliography_is_processed = False
    release_information_is_collected = False

    bold_counter = 0
    for sel in text_content:
        last_elem_type = get_elem_type(sel)
        if re.search('Участники{,1}:', process(sel)) is not None:
            introduction_is_processed = False
        if ':' in process(sel) and re.search('Участники{,1}:', process(sel)) is None:
            participants_are_processed = False
        if last_elem_type == 'b':
            if sel.xpath('text()').extract() == 'Библиография':
                text_is_processed = False
        if bibliography_is_processed and last_elem_type == 'b':
            bibliography_is_processed = False
            release_information_is_collected = True
        if release_information_is_collected and last_elem_type != 'b':
            release_information_is_collected = False 

        if introduction_is_processed:
            introduction += process(sel)
        if participants_are_collected:
            string = process(sel)
            if len(string) > 0:
                if string != '\n':
                    participants.append(string)
        if text_is_processed:
            text += process(sel)
        if bibliography_is_processed:
            text = process(sel)
            if len(text) > 0:
                if text != '\n':
                    bibliography += text + '\n'
        if release_information_is_collected:
           release_information += process(sel)

        if re.search('Участники{,1}:', process(sel)) is not None and len(text) == 0:
            participants_are_processed = True
        if ':' in process(sel) and re.search('Участники{,1}:', process(sel)) is None:
            text_is_processed = True
        if last_elem_type == 'b':
            if sel.xpath('text()').extract() == 'Библиография':
                bibliography_is_processed = True
    return introduction, participants, text, bibliography, release_information


def parse_paragraph(sel, current_speaker):
    text = ''
    part_sels = sel.xpath('b | text() | i | blockquote | nobr')
    if get_elem_type(part_sels[0]) == 'b':
        
    for part_sel in part_sels:
        i
        


class GordonSpider(scrapy.Spider):
    name = "gordon_spider"

    def start_requests(self):
        urls = ['http://gordon0030.narod.ru/index.html']
        for url in urls:
            yield scrapy.Request(url=url, callback=self._archive_page)

    def _archive_page(self, response):
        for link_selector in response.css('body>table>tr>td>div>ol>li'):
            refs = link_selector.css('a::attr(href)').extract()
            #print('refs:', refs)
            if len(refs) > 0:
                if len(refs[0]) > 0:
                    if refs[0][0] != '#':
                        url = response.urljoin(refs[0])
                        print('url:', url)
                        request = scrapy.Request(url, callback=self._issue_page)
                        request.meta['issue_name'] = response.css('body>table>tr>td>div>ol>li>a::text').extract_first()
                        yield request

    def _issue_page(self, response):
        if 'issue_name' in response.meta:
            issue_name = response.meta['issue_name']
        else:
            issue_name = 'unknown'
        print('-'*20)
        print("parsing '%s'" % issue_name)
        print('-'*20)
        td = response.xpath('//body/table/tr/td[a[text()="Стенограмма эфира"]]')
        text_content = td.xpath('text() | b | nobr | blockquote')
        introduction, participants, text, bibliography, release_information = extract_text_and_additional_information(text_content)
        folder_name = add_index_to_filename_if_needed(os.path.expanduser("~") + '/gordon/gordon_result/' + 'issue_name')
        create_path(folder_name)
        with open(folder_name + '/introduction.txt', 'w') as f:
            f.write(introduction)
        with open(folder_name + '/participants.txt', 'w') as f:
            for participant in participants:
                f.write(participant + '\n')
        with open(folder_name + '/materials.txt', 'w') as f:
            f.write(text)
        with open(folder_name + '/bibliography.txt', 'w') as f:
            f.write(bibliography)
        with open(folder_name + '/release_inf.txt', 'w') as f:
            f.write(release_information)

        urls = td.xpath('a/@href').extract()
        if len(urls) == 0:
            print('Error: link is not provided. Issue %s' % issue_name)
        else:
            url = urls[0]
        yield scrapy.Request(url, callback=self._transcript_page)

    def _transcript_page(self, response):
        par_sels = response.xpath('//body/p')
         
        




