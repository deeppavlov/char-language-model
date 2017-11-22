import re
import os

from collections import OrderedDict

uppercase_russian = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЫЪЭЮЯ"
lowercase_russian = "абвгдеёжзийклмнопрстуфхцчшщьыъэюя"
russian = uppercase_russian + lowercase_russian
latin = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
punctuation = "'\",\\.;: "
russian_latin_punctuation = russian + latin + punctuation

fname_and_lname = re.compile('[%s][%s]+ [%s][%s]+' % (uppercase_russian, lowercase_russian, uppercase_russian, lowercase_russian))
initials = re.compile('[%s]\.[%s]\.' % (uppercase_russian, uppercase_russian))


def check_type_layout(sels, year, month):
    types = list()
    for sel in sels:
        types.append(get_elem_type(sel))
    if len(types) % 2 != 0:
        raise UnexpectedElementsError('Number of h1 and div elements should be even (year: %s, month: %s)' % (year, month),
                                      len(types),
                                      types,
                                      '[\'h1\', \'div\']*2k',
                                      year,
                                      month)
    else:
        correct_types = ['h1', 'div'] * (len(types) // 2)
        if types != correct_types:
            raise UnexpectedElementsError('h1 and div elements should interchange (year: %s, month: %s)' % (year, month),
                                      len(types),
                                      types,
                                      correct_types,
                                      year,
                                      month)


def construct(obj):
    """Used for preventing of not expected changing of class attributes"""
    if isinstance(obj, OrderedDict):
        new_obj = OrderedDict()
        for key, value in obj.items():
            new_obj[key] = construct(value)
    elif not isinstance(obj, OrderedDict) and isinstance(obj, dict):
        new_obj = dict()
        for key, value in obj.items():
            new_obj[key] = construct(value)
    elif isinstance(obj, list):
        new_obj = list()
        for value in obj:
            new_obj.append(construct(value))
    elif isinstance(obj, tuple):
        base = list()
        for value in obj:
            base.append(construct(value))
        new_obj = tuple(base)
    elif isinstance(obj, str):
        new_obj = str(obj)
    elif isinstance(obj, (int, float, complex, type(None))) or inspect.isclass(obj):
        new_obj = obj
    else:
        raise TypeError("Object of unsupported type was passed to construct function: %s" % type(obj))
    return new_obj


def fix_similar_looking_latin(string):
    string = re.sub('A', 'А', string)
    string = re.sub('B', 'В', string)
    string = re.sub('C', 'С', string)
    string = re.sub('E', 'Е', string)
    string = re.sub('H', 'Н', string)
    string = re.sub('K', 'К', string)
    string = re.sub('M', 'М', string)
    string = re.sub('O', 'О', string)
    string = re.sub('P', 'Р', string)
    string = re.sub('T', 'Т', string)
    string = re.sub('X', 'Х', string)
    string = re.sub('a', 'а', string)
    string = re.sub('c', 'с', string)
    string = re.sub('e', 'е', string)
    string = re.sub('o', 'о', string)
    string = re.sub('p', 'р', string)
    string = re.sub('x', 'х', string)
    return string


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


def f_text(text):
    text = re.sub('\\xa0', ' ', text)
    text = re.sub('&nbsp;', ' ', text)
    text = re.sub(' {2,}', ' ', text)
    return text


def get_elem_type(sel):
    extr = sel.extract()
    if len(extr) == 0:
        el_type = None
    elif extr[0] == '<':
        m = re.search('<([^> ]+)', extr)
        el_type = m.group(1)
    else:
        el_type = 'text'
    return el_type


def process_simple_text(sel):
    return ' ' + f_text(''.join(sel.extract())) + ' '


def process_nobr(sel):
    return ' ' + f_text(''.join(sel.xpath('text()').extract())) + ' '


def process_bold(bold_sel):
    return '<b>' + f_text(''.join(bold_sel.xpath('text()').extract())) + '</b>'


def process_italic(italic_sel):
    return '<i>' + f_text(''.join(italic_sel.xpath('text()').extract())) + '</i>'


def process_quote(quote_sel):
    text = ''
    sels = quote_sel.xpath('text() | i | b | nobr | blockquote')
    for sel in sels:
        text += process(sel)
    text = re.sub(' {2,}', ' ', text)
    return text


def process(sel):
    mode = get_elem_type(sel)
    # print('(process)mode:', mode)
    text = '' 
    if mode == 'text':
        text += process_simple_text(sel)
    elif mode == 'b':
        text += process_bold(sel)
    elif mode == 'i':
        text += process_italic(sel)
    elif mode == 'blockquote':
        text += process_quote(sel)
    elif mode == 'nobr':
        text += process_nobr(sel)
    # print('(process)text:', text)
    return text


def filter_text(text, remove_separating_chars, remove_leading_and_trailing):
    text = re.sub('<[^>]*>', ' ', text)
    if remove_separating_chars:
        text = re.sub('(?<!^)[\n\t\r]+(?!$)', ' ', text)
    text = re.sub(' {2,}', ' ', text) 
    if remove_leading_and_trailing:
        text = re.sub('[ \t\n\r]+$', '', text)
        text = re.sub('^[ \t\n\r:\.]+', '', text)
    return text    


def extract_text_and_additional_information(text_content, issue_name):
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

    source_name = ''
    collecting_source_name = False

    bold_counter = 0
    for sel in text_content:
        last_elem_type = get_elem_type(sel)
        s = process(sel)
        if last_elem_type == 'b' and len(text) == 0 and len(participants) == 0:
            introduction_is_processed = False
        if last_elem_type == 'b':
            if re.search("[Оо]бзор", s) is not None or re.search("[Мм]атериалы", s) is not None:
                participants_are_processed = False
        if last_elem_type == 'b':
            t = s[3:-4]
            # print('bold:', t)
            if t == 'Библиография:' or t == 'Библиография' or t == 'библиография' or t == 'библиография:':
                text_is_processed = False
        if bibliography_is_processed and last_elem_type == 'b':
            bibliography_is_processed = False
            release_information_is_collected = True
        if release_information_is_collected and last_elem_type != 'b':
            release_information_is_collected = False 

        if introduction_is_processed:
            s = filter_text(s, True, True)
            if len(introduction) > 0:
                introduction += ' '
            introduction += s
        if participants_are_processed:
            if len(s) > 0:
                if s != '\n':
                    if last_elem_type == 'b':
                        s = filter_text(s, True, True)
                        participants.append(s)
        if text_is_processed:
            text += s
        if bibliography_is_processed:
            # print('bibliography_is_processed')
            # print('s:', s)
            if last_elem_type == 'br':
                # print('br,', source_name)
                collecting_source_name = False
                if len(source_name) > 0:
                    # print('len(source_name) > 0')
                    source_name = filter_text(source_name, True, True)
                    # print('after filtering:', source_name)
                    if len(source_name) > 0:
                        # print('!adding to bibliography')
                        bibliography += source_name + '\n'
                source_name = ''
            else:
                # print('not br')
                collecting_source_name = True
            if collecting_source_name:
                # print('adding string:', s)
                source_name += s
                # print('source name modified:', source_name)
            

        if release_information_is_collected:
           if len(release_information) > 0:
               release_information += '\n'
           release_information += s

        if re.search('Участники{,1}:', s) is not None and len(text) == 0:
            participants_are_processed = True
        # if ':' in s and re.search('Участники{,1}:', s) is None:
        #     text_is_processed = True
        if last_elem_type == 'b' and len(text) == 0:
            if re.search("[Оо]бзор", s) is not None or re.search("[Мм]атериалы", s) is not None:
                text_is_processed = True
        if last_elem_type == 'b':
            t = s[3:-4]
            # print('bold:', t)
            if t == 'Библиография:' or t == 'Библиография' or t == 'библиография' or t == 'библиография:':
                bibliography_is_processed = True

    introduction = filter_text(introduction, False, True)
    bibliography = filter_text(bibliography, False, True)
    release_information = filter_text(release_information, False, True)
    release_information = re.sub(' *[\n]+ *', '\n', release_information)

    # if len(introduction) == 0:
    #     print("'%s': no introduction" % issue_name)
    # else:
    #     print("'%s': introduction is present" % issue_name)

    # if len(participants) == 0:
    #     print("'%s': no participants" % issue_name)
    # else:
    #     print("'%s': participants section is present" % issue_name)

    # if len(text) == 0:
    #     print("'%s': no text" % issue_name)
    # else:
    #     print("'%s': text is present" % issue_name)

    # if len(bibliography) == 0:
    #     print("'%s': no bibliography" % issue_name)
    # else:
    #     print("'%s': bibliography is present" % issue_name)

    # if len(release_information) == 0:
    #     print("'%s': no release information" % issue_name)
    # else:
    #     print("'%s': release information is present" % issue_name)
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
    if idx > 0:
        sels = sels[idx-1:]
    return speakers, sels


def prepair_speaker_name_abbr(speaker_name):
    speaker_name = re.sub(':', '', speaker_name)
    speaker_name = fix_similar_looking_latin(speaker_name)
    speaker_name = re.sub('[^%s]' % russian_latin_punctuation, '', speaker_name)
    if ' ' in speaker_name:
        speaker_name = speaker_name.split()
        first_res = list()
        for part in speaker_name:
            if len(part) > 0:
                if part[-1] == '.':
                    part = part[:-1]
                first_res.append(part)
    else:
        first_res = [speaker_name]
    second_res = list()
    for one_res in first_res:
        if '.' in one_res:
            one_res = one_res.split('.')
            for part in one_res:
                if len(part) > 0:
                    second_res.append(part)
        else:
            second_res.append(one_res)
    return second_res    


def names_match(tpl, speaker_name):
    # print('speaker_name:', speaker_name)
    speaker_name = prepair_speaker_name_abbr(speaker_name)
    # print('(after prepairing)speaker_name:', speaker_name)
    orig_list = list(tpl)
    cand_list = list(speaker_name)
    if len(cand_list) > len(orig_list):
        match = False
    else:
        match = True
        for cand_part in cand_list:
            cand_part_match = False
            o = list(orig_list)
            for orig_idx, orig_part in enumerate(o):
                if len(cand_part) <= len(orig_part):
            
                    if cand_part == orig_part[:len(cand_part)] and not cand_part_match:
                        cand_part_match = True
                        o_tmp = o[:orig_idx]
                        o_tmp.extend(o[orig_idx+1:])
                        o = o_tmp

            match = match and cand_part_match
    return match
                

def get_matches(speaker, names):
    matches = dict()
    for name, speaker_idx in names.items():
        if names_match(name, speaker):
            matches[name] = speaker_idx
    return matches


def process_speaker_new_name(abbr, speakers, issue_name, base_folder):
    matches = get_matches(abbr, speakers['original_names'])
    if len(matches) == 0:
        speaker_idx = 1
        speakers['map'][abbr] = speaker_idx
        with open(base_folder + '/logs.txt', 'a') as f:
            f.write("error: no matches found for abbreviation '%s' in issue '%s', speakers: '%s'\n\n" % (abbr, issue_name, str(speakers['original_names'])))
    elif len(matches) == 1:
        speakers['map'][abbr] = list(matches.values())[0]
        speaker_idx = list(matches.values())[0]
    elif len(matches) > 1:
        if len(matches) == 2:
            if ('Александр', 'Гордон') in matches:
                del matches[('Александр', 'Гордон')]
                speaker_idx = list(matches.values())[0]
                speakers['map'][abbr] = speaker_idx
                with open(base_folder + '/logs.txt', 'a') as f:
                    # print('abbr:', abbr)
                    # print('issue_name:', abbr)
                    # print("str(speakers['original_names']):", str(speakers['original_names']))
                    # print('list(matches.keys())[0]:', list(matches.keys())[0])
                    f.write("error: 2 matches found for abbreviation '%s' in issue '%s', speakers: '%s'\n\t" % (abbr, issue_name, str(speakers['original_names'])) + \
                            "'%s' speaker is chosen\n\n" % str(list(matches.keys())[0]))
            else:
                speaker_idx = list(matches.values())[0]
                speakers['map'][abbr] = speaker_idx
                with open(base_folder + '/logs.txt', 'a') as f:
                    f.write("error: 2 matches found for abbreviation '%s' in issue '%s', speakers: '%s'\n\t" % (abbr, issue_name, str(speakers['original_names'])) + \
                            "'%s' speaker is chosen (conflict is resolved randomly)\n\n" % str(list(matches.keys())[0]))
        else:
            if ('Александр', 'Гордон') in matches:
                del matches[('Александр', 'Гордон')]
                speaker_idx = list(matches.values())[0]
                speakers['map'][abbr] = speaker_idx
                with open(base_folder + '/logs.txt', 'a') as f:
                    f.write("error: %s matches found for abbreviation '%s' in issue '%s', speakers: '%s'\n\t" % (len(matches) + 1, abbr, issue_name, str(speakers['original_names'])) + \
                            "'%s' speaker is chosen\n\n" % str(list(matches.keys())[0]))
            else:
                speaker_idx = list(matches.values())[0]
                speakers['map'][abbr] = speaker_idx
                with open(base_folder + '/logs.txt', 'a') as f:
                    f.write("error: %s matches found for abbreviation '%s' in issue '%s', speakers: '%s'\n\t" % (len(matches), abbr, issue_name, str(speakers['original_names'])) + \
                            "'%s' speaker is chosen (conflict is resolved randomly)\n\n" % str(list(matches.keys())[0]))     
    return speaker_idx, speakers                       


def parse_paragraph(sel, fd, current_speaker, speakers, issue_name, base_folder):
    speakers = construct(speakers)
    text = ''
    part_sels = sel.xpath('b | text() | i | blockquote | nobr')
    # print("('%s') len(part_sels):" % issue_name, len(part_sels))
    if get_elem_type(part_sels[0]) == 'b':
        speaker = part_sels[0].xpath('text()').extract()
        if len(speaker) > 0:
            if len(speaker[0]) > 0:
                speaker = speaker[0]
            else:
                speaker = current_speaker
        else:
            speaker = current_speaker
        text_sels = part_sels[1:]
    else:
        speaker = current_speaker
        text_sels = part_sels
    if speaker in speakers['map']:
        speaker_idx = speakers['map'][speaker]
    else:
        speaker_idx, speakers = process_speaker_new_name(speaker, speakers, issue_name, base_folder)
    # print('speakers:', speakers)
    if current_speaker in speakers['map']:
        current_speaker_idx = speakers['map'][current_speaker]
    else:
        current_speaker_idx, speakers = process_speaker_new_name(current_speaker, speakers, issue_name, base_folder)

    for sel in text_sels:
        # part_sels = sel.xpath('b | text() | i | blockquote | nobr')
        # for part_sel in part_sels:
        text += process(sel)
    text = filter_text(text, True, True)
    # print('(parse_paragraph)len(text):', len(text)) 
    if len(text) > 0:
        if speaker_idx != current_speaker_idx: 
            if fd.tell() > 0:
                fd.write('\n')
            text = '<%s>' % speaker_idx + text   
        # print('ready to write')  
        fd.write(text)
    return speaker, speakers


def divide_into_issues(sels):
    issues = list()
    for i in range(len(sels) // 2):
        issues.append((sels[2 * i], sels[2 * i + 1]))
    return issues


def retrieve_speakers0030(sels):
    speakers = list()
    speakers_par = False
    idx = 0
    while not speakers_par and idx < len(sels):
        s = process(sels[idx])
        if re.search('Участники{,1}', s) is not None and len(speakers) == 0:
            speakers_par = True
        idx += 1
    while speakers_par and idx < len(sels):
        if get_elem_type(sels[idx]) == 'i':
            speakers.append(filter_text(process(sels[idx]), True, True))
        if speakers_par:
            if get_elem_type(sels[idx]) == 'b':
                speakers_par = False
        idx += 1
    if idx > 0:
        sels = sels[idx-1:]
    return speakers, sels


def selector_is_speaker_abbr(sel):
    if get_elem_type(sel) == 'b':
        extr = sel.xpath('text()').extract()
        if len(extr) > 0:
            if len(extr) > 1:
                extr = ' '.join(extr)
            else:
                extr = extr[0]
            if fname_and_lname.search(extr) is not None or fname_and_lname.search(extr) is not None:
                return True
            else:
                return False
        else:
            return False
    return False
    

def parse0030(sels, issue_name, speaker_list, base_folder):
    current_speaker_idx = -1
    start_of_replica = True
    speakers = {'original_names': {('Александр', 'Гордон'): 0},
                'map': {'Александр Гордон': 0,
                        'А.Г.': 0,
                        'А. Г.': 0,
                        'Гордон Александр': 0,
                        '': -1}}
    for speaker_idx, speaker in enumerate(speaker_list):
        speakers['original_names'][tuple(speaker.split())] = speaker_idx + 1
    transcript = ''
    for sel in sels:
        s = process(sel)
        s = filter_text(s, True, True)
        s = fix_similar_looking_latin(s)
        if selector_is_speaker_abbr(sel):
            if len(s) > 0:
                if s not in speakers['map']:
                    speaker_idx, speakers = process_speaker_new_name(s, speakers, issue_name, base_folder)
                else:
                    speaker_idx = speakers['map'][s]
            if speaker_idx != current_speaker_idx:
                if len(transcript) > 0:
                    transcript += '\n' 
                transcript += '<%s>' % speaker_idx
                start_of_replica = True
                current_speaker_idx = speaker_idx
        else:
            if not start_of_replica:
                transcript += ' ' + s
            else:
                transcript += s
                start_of_replica = False
    return transcript


class UnexpectedElementsError(Exception):
    def __init__(self, msg, length, types, required_types, year, month):
        inp_inf_str = ''
        inp_inf_str += '\nlength = %s' % length + '\ntypes = %s' % str(types) + '\nrequired_types = %s' % str(required_types) 
        self._msg = msg
        self._length = length
        self._types = types
        self._required_types = required_types
        self._year = year
        self._month = month
        super(UnexpectedElementsError, self).__init__(msg + inp_inf_str)


def check_if_there_is_sense(s):
    if re.search('[%s]+' % (russian + latin), s) is not None:
        return True
    else:
        return False


def select_sensible_strings(strings):
    sensible_strings = list()
    for string in strings:
        if check_if_there_is_sense(string):
            sensible_strings.append(string)
    return sensible_strings


def divide_into_issues_marsexx(sels):
    issues = list()
    issue = list()
    for sel in sels:
        if len(issue) > 0 and get_elem_type(sel) == 'h1':
            issues.append(issue)
            issue = [sel]
        else:
            issue.append(sel)
    return issues


def retrieve_speakers_marsexx(pars):
    # print('\n')
    speakers = list()
    idx = 0
    while idx < len(pars) and len(pars[idx].xpath('@class')) > 0:
        extr = pars[idx].xpath('b/text()').extract()
        # print('extracted text:', extr)
        extr = ''.join(extr)
        extr = filter_text(extr, True, True)
        speakers.append(extr)
        idx += 1
    return speakers, pars[idx:]


def expand(perms, remaining):
    new_ps = list()
    new_rs = list()
    for p, r in zip(perms, remaining):
        for el_idx, el in enumerate(r):
            new_p = construct(p)
            for perm in new_p:
                perm.append(el)
            new_r = construct(r[:el_idx])
            new_r.extend(construct(r[el_idx + 1:]))
            new_ps.append(new_p)
            new_rs.append(new_r)
    return new_ps, new_rs

def create_permutations(lst):
    perms = [[[]]]
    remaining = [construct(lst)]
    for _ in range(len(lst)):
        perms, remaining = expand(perms, remaining)
    res = list()
    for perm in perms:
        for p in perm:
            res.append(p)
    return res