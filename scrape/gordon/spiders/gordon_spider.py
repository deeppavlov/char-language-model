import scrapy
import os

from help_functions import (extract_text_and_additional_information, add_index_to_filename_if_needed,
                            create_path, retrieve_speakers, parse_paragraph)


class GordonSpider(scrapy.Spider):
    name = "gordon_spider"
    _base_folder = os.path.expanduser("~") + '/gordon/gordon_result/'

    def start_requests(self):
        urls = ['http://gordon0030.narod.ru/index.html']
        for url in urls:
            yield scrapy.Request(url=url, callback=self._archive_page)

    def _archive_page(self, response):
        for link_selector in response.xpath('//body/table/tr/td/div/ol/li')[2:]:
            refs = link_selector.xpath('a/@href').extract()
            #print('refs:', refs)
            if len(refs) > 0:
                if len(refs[0]) > 0:
                    if refs[0][0] != '#':
                        url = response.urljoin(refs[0])
                        # print('(on archive page) url:', url)
                        request = scrapy.Request(url, callback=self._issue_page)
                        request.meta['issue_name'] = link_selector.xpath('a/text()').extract_first(default='unknown')
                        yield request

    def _issue_page(self, response):
        if 'issue_name' in response.meta:
            issue_name = response.meta['issue_name']
        else:
            issue_name = 'unknown'
        print('-'*20)
        print("parsing '%s'" % issue_name)
        print('-'*20)
        td = response.xpath('//body/table/tr[8]/td')
        # td = response.xpath('//body/table/tr/td[b[contains(text(), "Хронометраж")] | b[contains(text(), "Тема №")] | b[contains(text(), "Эфир")]]')
        if len(td) == 0:
            print("can't get td. issue '%s'" % issue_name)
        # td = response.xpath('//body/table/tr/td[a[text()="Стенограмма эфира"]]')
        text_content = td.xpath('text() | b | nobr | blockquote | i | br')
        # print('text_content:', text_content)
        introduction, participants, text, bibliography, release_information = extract_text_and_additional_information(text_content, issue_name)
        # print('_base_folder:', self._base_folder)
        folder_name = add_index_to_filename_if_needed(self._base_folder + '/' + issue_name)
        create_path(folder_name)
        
        if len(introduction) > 0:
            with open(folder_name + '/introduction.txt', 'w') as f:
                f.write(introduction)

        if len(participants)> 0:
            with open(folder_name + '/participants.txt', 'w') as f:
                for participant in participants:
                    f.write(participant + '\n')
        # text = filter_text(text, False, True)
        if len(text)> 0:
            with open(folder_name + '/materials.txt', 'w') as f:
                f.write(text)
        
        if len(bibliography) > 0:
            with open(folder_name + '/bibliography.txt', 'w') as f:
                f.write(bibliography)
        
        if len(release_information) > 0:
            with open(folder_name + '/release_inf.txt', 'w') as f:
                f.write(release_information)

        urls = td.xpath('a/@href').extract()
        if len(urls) == 0:
            print('Error: link is not provided. Issue %s' % issue_name)
        else:
            url = response.urljoin(urls[0])
            print('(on issue page) url:', url)
            request = scrapy.Request(url, callback=self._transcript_page)
            request.meta['folder_name'] = folder_name 
            request.meta['issue_name'] = issue_name
            yield request

    def _transcript_page(self, response):
        folder_name = response.meta['folder_name']
        issue_name = response.meta['issue_name']
        # print('(on trascript page) issue_name:', issue_name)
        transcript_file_name = folder_name + '/transcript.txt'
        trascript_roles_file_name = folder_name + '/trascript_roles.txt'
        par_sels = response.xpath('//body/p')
        # print("('%s') len(par_sels):" % issue_name, len(par_sels))
        speaker_list, par_sels = retrieve_speakers(par_sels)
        # print("('%s') speaker list:" % issue_name, speaker_list)
        # print("('%s') len(par_sels):" % issue_name, len(par_sels))
        speakers = {'original_names': {('Александр', 'Гордон'): 0},
                    'map': {'Александр Гордон': 0,
                            'А.Г.': 0,
                            'А. Г.': 0,
                            'Гордон Александр': 0,
                            '': -1}}
        for speaker_idx, speaker in enumerate(speaker_list):
            speakers['original_names'][tuple(speaker.split())] = speaker_idx + 1

        with open(trascript_roles_file_name, 'w') as fd:
            for name, speaker_idx in speakers['original_names'].items():
                fd.write(str(speaker_idx) + ':')
                for word in name:
                    fd.write(' ' + word)
                fd.write('\n')
        
        current_speaker = ''
        if len(par_sels) > 0:
            with open(transcript_file_name, 'w') as fd:
                for sel in par_sels:
                    current_speaker, speakers = parse_paragraph(sel, fd, current_speaker, speakers, issue_name, self._base_folder)
            
         
        




