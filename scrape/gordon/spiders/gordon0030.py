import scrapy
import os
from help_functions import (add_index_to_filename_if_needed, create_path, check_type_layout, divide_into_issues,
                            retrieve_speakers0030, parse0030, get_elem_type, select_sensible_strings, filter_text)

class Gordon0030(scrapy.Spider):
    name = "gordon0030"
    _base_folder = os.path.expanduser("~") + '/gordon/gordon0030/'

    def _log(self, msg):
        with open(self._base_folder + 'logs.txt', 'a') as f:
            f.write(msg)        

    def _get_issue_name(self, header_el, year, month, idx):
        names = header_el.xpath('text()').extract()
        names = select_sensible_strings(names)
        if len(names) == 0:
            self._log('Error! no issue was found in header %s (year: %s, month: %s)' % (idx, year, month))
            name = 'unknown'
        elif len(names) > 1:
            self._log('Error! %s issue names were found in header %s (year: %s, month: %s)' % (len(names), idx, year, month))
            name = filter_text(names[0], True, True)
        else:
            name = filter_text(names[0], True, True)
        return name    

    def start_requests(self):
        tmpls = dict([(i, 'http://gordon0030.narod.ru/transcripts/200%s' % i + '-%s.html') for i in range(1,4)])
        urls = dict()
        for year, tmpl in tmpls.items():
            for month in range(1, 13):
                if month >= 10:
                    urls[(year, month)] = tmpl % month
                else:
                    urls[(year, month)] = tmpl % ('0' + str(month))
        create_path(self._base_folder)
        for (year, month), url in urls.items():
            request = scrapy.Request(url=url, callback=self._month_page)
            request.meta['year'] = year
            request.meta['month'] = month
            yield request

    def _month_page(self, response):
        relevant = response.xpath('//body/li')
        headers_and_content = relevant.xpath('div[not(@align="center")] | h1[@align="center"] |'
                                             ' h4[text()="Сноски"]')
        idx = 0
        snoska_idx = None
        while snoska_idx is None and idx < len(headers_and_content):
            if get_elem_type(headers_and_content[idx]) == 'h4':
                snoska_idx = idx
            idx += 1
        # print(snoska_idx)
        if snoska_idx is not None:
            headers_and_content = headers_and_content[:snoska_idx]
        year = response.meta['year']
        month = response.meta['month']
        check_type_layout(headers_and_content, year, month)
        issues = divide_into_issues(headers_and_content)
        for idx, (header, content) in enumerate(issues):
            issue_name = self._get_issue_name(header, year, month, idx)
            text_nodes = content.xpath('text() | b | i | nobr | blockquote')
            speaker_list, sels = retrieve_speakers0030(text_nodes)
            print('issue_name:', issue_name, 'len(sels):', len(sels), 'len(text_nodes):', len(text_nodes),
                  'speakers:', speaker_list, 'content:', content)
            folder_name = self._base_folder + issue_name
            folder_name = add_index_to_filename_if_needed(folder_name)
            create_path(folder_name)
            transcript = parse0030(sels, issue_name, speaker_list, self._base_folder)
            if len(speaker_list) > 0:
                with open(folder_name + '/transcript_roles.txt', 'w') as f:
                    f.write('0: ' + 'Александр Гордон')
                    for speaker_idx, speaker in enumerate(speaker_list):
                        f.write(('\n%s: ' % (speaker_idx + 1)) + speaker) 
            if len(transcript) > 0:
                with open(folder_name + '/transcript.txt', 'w') as f:
                    f.write(transcript)
            else:
                msg = 'Error! Zero length transcript.\nYear: %s, Month: %s, Issue name: %s' % (year, month, issue_name)
                self._log(msg)
                print(msg)
            
       
         
        

