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
    text = re.sub(' {2,}', ' ', text)
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
    text = re.sub(' {2,}', ' ', text)
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
        if participants_are_processed:
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


def retrieve_speakers(sels):
    speakers = list()
    speakers_par = False
    idx = 0
    while not speakers_par and idx < len(sels):
        sel = sels[idx]
        btexts = sel.xpath('b/text()').extract()
        if len(btexts) > 0:
            if re.search('Участники{,1}:', btexts[0]) is not None:
                speakers_par = True
        idx += 1
    while speakers_par and idx < len(sels):
        btexts = sels[idx].xpath('b/text()').extract()
        if len(btexts) > 0:
            if ':' in btexts[0]:
                speakers_par = False
        if speakers_par:
            italic_texts = sels[idx].xpath('i/text()').extract()
            if len(italic_texts) > 0:
                if len(italic_texts[0]) > 0:
                    speakers.append(italic_texts[0])
        idx += 1
    return speakers, sels[idx:]


def prepair_speaker_name_abbr(speaker_name):
    if ' ' in speaker_name:
        speaker_name = speaker_name.split()
        first_res = list()
        for part in speaker_name:
            if len(part) > 0:
                if part[-1] == '.':
                    part = part[:-1]
                first_res.append(part)
    second_res = list()
    for one_res in first_res:
        if '.' in one_res:
            one_res = one_res.split('.')
            for part in one_res:
                if len(part) > 0:
                    second_res.append(part)
    return second_res    


def names_match(tpl, speaker_name):
    speaker_name = prepair_speaker_name_abbr(speaker_name)
    orig_list = list(tpl)
    cand_list = list(speaker_name)
    if len(cand_list) > len(orig_list):
        match = False
    else:
        match = True
        for cand_part in cand_list:
            cand_part_match = False
            for orig_idx, orig_part in enumerate(orig_list):
                if cand_part == orig_part[:len(cand_part)] and not cand_part_match:
                    cand_part_match = True
                    o = orig_list[:orig_idx]
                    o.extend(orig_list[orig_idx+1:])
            match = match and cand_part_match
    return match
                

def get_matches(speaker, names):
    matches = dict()
    for name, speaker_idx in names.items():
        if names_match(name, speaker):
            matches[name] = speaker_idx
    return matches


def process_speaker_new_name(abbr, speakers, issue_name):
    matches = get_matches(abbr, speakers['original_names'])
    if len(matches) == 0:
        speaker_idx = 1
        speakers['map'][abbr] = speaker_idx
        with open(self._base_folder + '/logs.txt', 'a') as f:
            f.write("error: no matches found for abbreviation '%s' in issue '%s', speakers: '%s'\n\n" % (abbr, issue_name, str(speakers['original_names'])))
    elif len(matches) == 1:
        speakers['map'][abbr] = list(matches.values())[0]
    elif len(matches) > 1:
        if len(matches) == 2:
            if ('Александр', 'Гордон') in matches:
                del matches[('Александр', 'Гордон')]
                speaker_idx = list(matches.values())[0]
                speakers['map'][abbr] = speaker_idx
                with open(self._base_folder + '/logs.txt', 'a') as f:
                    f.write("error: 2 matches found for abbreviation '%s' in issue '%s', speakers: '%s'\n\t" % (abbr, issue_name, str(speakers['original_names'])) + \
                            "'%s' speaker is chosen\n\n" % list(matches.keys())[0])
            else:
                speaker_idx = list(matches.values())[0]
                speakers['map'][abbr] = speaker_idx
                with open(self._base_folder + '/logs.txt', 'a') as f:
                    f.write("error: 2 matches found for abbreviation '%s' in issue '%s', speakers: '%s'\n\t" % (abbr, issue_name, str(speakers['original_names'])) + \
                            "'%s' speaker is chosen (conflict is resolved randomly)\n\n" % list(matches.keys())[0])
        else:
            if ('Александр', 'Гордон') in matches:
                del matches[('Александр', 'Гордон')]
                speaker_idx = list(matches.values())[0]
                speakers['map'][abbr] = speaker_idx
                with open(self._base_folder + '/logs.txt', 'a') as f:
                    f.write("error: %s matches found for abbreviation '%s' in issue '%s', speakers: '%s'\n\t" % (len(matches) + 1, abbr, issue_name, str(speakers['original_names'])) + \
                            "'%s' speaker is chosen\n\n" % list(matches.keys())[0])
            else:
                speaker_idx = list(matches.values())[0]
                speakers['map'][abbr] = speaker_idx
                with open(self._base_folder + '/logs.txt', 'a') as f:
                    f.write("error: %s matches found for abbreviation '%s' in issue '%s', speakers: '%s'\n\t" % (len(matches), abbr, issue_name, str(speakers['original_names'])) + \
                            "'%s' speaker is chosen (conflict is resolved randomly)\n\n" % list(matches.keys())[0])     
    return speakers                       


def parse_paragraph(sel, fd, current_speaker, speakers, issue_name):
    text = ''
    part_sels = sel.xpath('b | text() | i | blockquote | nobr')
    if get_elem_type(part_sels[0]) == 'b':
        speaker = part_sels[0].xpath('text()').extract()
        if len(speaker) == 0:
            if len(speaker[0]) == 0:
                speaker = current_speaker
            else:
                speaker = speaker[0]
        else:
            speaker = current_speaker
        text_sels = part_sels[1:]
    else:
        speaker = current_speaker
        text_sels = part_sels
    if speaker in speakers['map']:
        speaker_idx = speakers['map'][speaker]
    else:
        speaker_idx, speakers = process_speaker_new_name(speaker, speakers)
    if current_speaker in speakers['map']:
        current_speaker_idx = speakers['map'][current_speaker]
    else:
        current_speaker_idx, speakers = process_speaker_new_name(current_speaker, speakers, issue_name)

    for sel in text_sels:
        part_sels = sel.xpath('b | text() | i | blockquote | nobr')
        for part_sel in part_sels:
            text += process(part_sel)
    
    text = re.sub('<[^>]*>', ' ', text)
    text = re.sub('[\n\t]', ' ', text)
    text = re.sub(' {2,}', ' ', text)
    if len(text) > 0:
        if text[0] == ' ':
            text = text[1:]
        if text[-1] == ' ':
            text = text[:-1]
        if speaker_idx != current_speaker_idx: 
            text = '<%s>' % speaker_idx + text     
        fd.write(text)
    return speaker, speakers


class GordonSpider(scrapy.Spider):
    name = "gordon_spider"
    _base_folder = os.path.expanduser("~") + '/gordon/gordon_result/'

    def start_requests(self):
        urls = ['http://gordon0030.narod.ru/index.html']
        for url in urls:
            yield scrapy.Request(url=url, callback=self._archive_page)

    def _archive_page(self, response):
        for link_selector in response.xpath('//body/table/tr/td/div/ol/li')[2:]:
            refs = link_selector.css('a::attr(href)').extract()
            #print('refs:', refs)
            if len(refs) > 0:
                if len(refs[0]) > 0:
                    if refs[0][0] != '#':
                        url = response.urljoin(refs[0])
                        print('url:', url)
                        request = scrapy.Request(url, callback=self._issue_page)
                        request.meta['issue_name'] = link_selector.xpath('a/@text').extract_first(default='unknown')
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
        # print('_base_folder:', self._base_folder)
        folder_name = add_index_to_filename_if_needed(self._base_folder + '/' + issue_name)
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
            request = scrapy.Request(url, callback=self._transcript_page)
            request.meta['folder_name'] = folder_name 
            request.meta['issue_name'] = issue_name
            yield request

    def _transcript_page(self, response):
        folder_name = response.meta['folder_name']
        issue_name = response.meta['issue_name']
        transcript_file_name = folder_name + '/transcript.txt'
        trascript_roles_file_name = folder_name + '/trascript_roles.txt'
        par_sels = response.xpath('//body/p')
        speaker_list, par_sels = retrieve_speakers(par_sels)
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
        
        with open(transcript_file_name, 'w') as fd:
            for sel in par_sels:
                parse_paragraph(sel, fd, current_speaker, speakers, issue_name)
            
         
        




