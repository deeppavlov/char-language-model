import scrapy
import os
from help_functions import (divide_into_issues_marsexx, add_index_to_filename_if_needed, retrieve_speakers_marsexx,
                            create_path, parse_paragraph, divide_into_issues_marsexx, filter_text)

class Marsexx(scrapy.Spider):
    name = "marsexx"
    _base_folder = os.path.expanduser("~") + '/gordon/marsexx/'

    def _log(self, msg):
        with open(self._base_folder + 'logs.txt', 'a') as f:
            f.write(msg)        

    def _get_issue_name(self, issue, idx):
        return issue[0].xpath('a/text()').extract_first(), issue[1:]  

    def start_requests(self):
        url = 'http://www.marsexx.ru/gordon-2.html'
        yield scrapy.Request(url=url, callback=self._site)

    def _site(self, response):
        headers_and_content = response.xpath('//body/table/tr/td/font/p[not(a/@href)] | //body/table/tr/td/font/h1')[2:-2]
        idx = 0
        issues = divide_into_issues_marsexx(headers_and_content)
        for idx, issue in enumerate(issues):
            print('issue name:', issue[0].xpath('a/text()').extract())
            issue_name, pars = self._get_issue_name(issue, idx)
            speaker_list, pars = retrieve_speakers_marsexx(pars)
            # print('issue_name:', issue_name, 'len(sels):', len(sels), 'len(text_nodes):', len(text_nodes), 'speakers:', speakers, 'content:', content)
            folder_name = self._base_folder + issue_name
            folder_name = add_index_to_filename_if_needed(folder_name)
            create_path(folder_name)
            transcript_file_name = folder_name + '/transcript.txt'
            trascript_roles_file_name = folder_name + '/transcript_roles.txt'
            speakers = {'original_names': {('Александр', 'Гордон'): 0},
                        'map': {'Александр Гордон': 0,
                                'А.Г.': 0,
                                'А. Г.': 0,
                                'Гордон Александр': 0,
                                '': -1}}
            with open(trascript_roles_file_name, 'w') as f:
                f.write('0: Александр Гордон')
                for speaker_idx, speaker in enumerate(speaker_list):
                    speakers['original_names'][tuple(speaker.split())] = speaker_idx + 1
                    f.write(('\n%s: ' % (speaker_idx + 1)) + speaker)

            current_speaker = ''
            if len(pars) > 0:
                with open(transcript_file_name, 'w') as fd:
                    for par in pars:
                        current_speaker, speakers = parse_paragraph(par, fd, current_speaker, speakers, issue_name, self._base_folder)
