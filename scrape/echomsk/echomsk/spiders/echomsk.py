import os
import re
import importlib.util
spec = importlib.util.spec_from_file_location("help", os.path.expanduser("~") +
                                              '/Natural-language-encoding/scrape/help_functions.py')
help = importlib.util.module_from_spec(spec)
spec.loader.exec_module(help)

import scrapy


class EchoMsk(scrapy.Spider):
    name = "echomsk"

    _log_limit = 2000000
    _total_char_limit = 3000000000000
    _limit_files = 300
    _std_inf = "\n\tissue_name: %s, page: %s, par_idx: %s, url: %s"
    _notifications = ['Notification 0: number of links on page %s: %s, url: %s',
                      'Notification 1: %s guest is added.\n\tIssue: \'%s\', page: %s, url: %s',
                      'Notification 2: %s leader is added.\n\tIssue: \'%s\', page: %s, url: %s',
                      "Notification 3: %s replicas found.\n\tIssue: \'%s\', page: %s, url: %s"]
    _warnings = ["Warning 0: bold element is not first."
                     "\n\tbold_elements: %s, first element type: %s, first element: %s" + _std_inf,
                 "Warning 1: more than one bold element.\n\tbold_elements: %s" + _std_inf,
                 "Warning 2: several_spans in paragraph." + _std_inf,
                 "Warning 3: number of bold elements is less than number of span elements. Elements: %s." + _std_inf,
                 "Warning 4: bold element is not followed by span element. Elements: %s." + _std_inf,
                 "Warning 5: found paragraph with several replicas. Elements: %s." + _std_inf,
                 "Warning 6: found uppecase speaker format. Speaker name: %s" + _std_inf,
                 "Warning 7: met speaker name containing 3 words. Speaker name: %s"
                     "\n\tissue_name: %s, page: %s, url: %s",
                 "Warning 8: Number of paragraphs is zero"
                     "\n\tissue_name: %s, page: %s, url: %s",
                 "Warning 9: both paragraphs and raw text are found. Number of paragraphs: %s, number of text nodes: %s"
                     "\n\tissue_name: %s, page: %s, url: %s"]
    _errors = ['Error 0: no issue name in reference element ---%s--- on page %s, url: %s',
               "Error 1: totally failed to get name for issue in reference element"
                   " ---%s--- on page %s.\n\t Falling to unknown. url: %s",
               'Error 2: no link in reference element ---%s--- on page %s, url: %s',
               "Error 3: failed to get leader name on issue \'%s\' on page %s, url: %s, Selector: %s",
               'Error 4: zero length name of leader.\n\tIssue: \'%s\', page: %s, url: %s',
               "Error 5: failed to get leader name on issue \'%s\' on page %s, url: %s. Selector: %s",
               'Error 6: zero length name of guest.\n\tIssue: \'%s\', page: %s, url: %s',
               "Error 7: leader name '%s' is found in speakers dictionary ---%s--- by"
                   " _check_if_in_speakers method but not found by get_matches functions."
                   "\n\tIssue: '%s', page: %s, url: %s",
               "Error 8: guest name '%s' is found in speakers dictionary ---%s--- by"
                    " _check_if_in_speakers method but not found by get_matches functions."
                    "\n\tIssue: '%s', page: %s, url: %s",
               "Error 9: more than 1 match found for leader '%s' in speakers dictionary (%s)."
                    " Falling to '%s'.\n\tIssue: '%s', page: %s",
               "Error 10: more than 1 match found for guest '%s' in speakers dictionary (%s)."
                   " Falling to '%s'.\n\tIssue: '%s', page: %s, url: %s",
               "Error 11: no matches found for abbreviation '%s', falling to '%s' speakers: '%s'"
                   "\n\tissue_name: %s, page_idx: %s, par_idx: %s, url: %s",
               "Error 12: %s matches for abbreviation '%s' found. Found abbreviations: %s. Speakers: %s"
                   "\n\tissue_name: %s, page_idx: %s, par_idx: %s, url: %s",
               "Error 13: paragraph is totally empty\n\tparagraph: %s" + _std_inf,
               "Error 14: paragraph is empty after stripping.\n\tparagraph: %s" + _std_inf,
               "Error 15: speaker syntax is provided but no speaker" + _std_inf,
               "Error 16: paragraph does not contain text nodes while speaker is provided" + _std_inf,
               "Error 17: no text.\n\tbold_elements: %s" + _std_inf,
               "Error 18: speaker descriptor is not provided." + _std_inf,
               "Error 19: Nor paragraphs or raw text found in issue '%s' on page %s (url: %s)",
               "Error 20: zero length reference for issue '%s', page: %s', url: %s",
               'Error 21: no relevant element on issue \'%s\' on page %s (url: %s)',
               'Error 22: no speaker block found on issue \'%s\' on page %s (url: %s)',
               'Error 23: no guest elements found on issue \'%s\' on page %s (url: %s)',
               'Error 24: no leader elements found on issue \'%s\' on page %s (url: %s)',
               'Error 25: No date and time were found. Page: %s, issue name: \'%s\'(url: %s)',
               'Error 26: no talk element on issue \'%s\' on page %s (url: %s)',
               "Error 27: bold element is empty" + _std_inf,
               "Error 28: element is bold but has incorrect format. Bold: %s" + _std_inf,
               "Error 29: empty replica" + _std_inf]
    _log_templates = help.unite_lists(_notifications, _warnings, _errors)

    @staticmethod
    def _form_default_speakers(names):
        speaker_names = names.split(', ')
        speakers_dict = {'original_names': {},
                         'map': {}}
        for idx, name in enumerate(speaker_names):
            if len(name) > 0:
                tpl = tuple(name.split())
                speakers_dict['original_names'][tpl] = str(idx) + 'в'
                speakers_dict['map'][name] = str(idx) + 'в'
                speakers_dict['map'][name.upper()] = str(idx) + 'в'
                speakers_dict['map'][tpl[0][0] + '. ' + tpl[1]] = str(idx) + 'в'
                speakers_dict['map'][(tpl[0][0] + '. ' + tpl[1]).upper()] = str(idx) + 'в'
        return speakers_dict

    def __init__(self,
                 start_url='https://echo.msk.ru/programs/naukafokus/',
                 num_pages=6,
                 res_folder_name='science_focus_res',
                 names='Асадова Наргиз, Быковский Егор',
                 *args, **kwargs):
        super(EchoMsk, self).__init__(*args, **kwargs)
        self._folder_index = 0
        self._log_index = 0
        self._col_chars = 0
        self._start_url = start_url
        self._num_pages = int(num_pages)
        if res_folder_name[0] == '/':
            self._base_folder = res_folder_name
        else:
            self._base_folder = os.path.expanduser("~") + '/Natural-language-encoding/scrape/echomsk/'\
                                + res_folder_name + '/'
        self._names = names
        self._default_speakers = self._form_default_speakers(self._names)

    def _save_logs_description(self):
        with open(self._base_folder + 'logs#%s.txt' % self._log_index, 'a') as f:
            f.write('----DESCRIPTION----\n')
            for tmpl in self._log_templates:
                f.write(tmpl + '\n')
            f.write('\n'*4 + '----LOGS:-----\n')

    def _log(self, msg):
        # print('logging')
        with open(self._base_folder + 'logs#%s.txt' % self._log_index, 'a') as f:
            f.write(msg + '\n')
            if f.tell() > self._log_limit:
                self._log_index += 1

    def _log_error(self, err_num, to_interpolate):
        try:
            msg = self._errors[err_num] % tuple(to_interpolate)
            self._log(msg)
        except TypeError:
            print('(_log_error) error number:', err_num)
            print('(_log_error) to_interpolate:', to_interpolate)
            print('(_log_error) self._errors[err_num]:', self._errors[err_num])
            raise

    def _log_warning(self, warn_num, to_interpolate):
        try:
            msg = self._warnings[warn_num] % tuple(to_interpolate)
            self._log(msg)
        except:
            print('(_log_warning) error number:', warn_num)
            print('(_log_warning) to_interpolate:', to_interpolate)
            print('(_log_warning) self._warnings[warn_num]:', self._warnings[warn_num])
            raise

    def _log_notification(self, notif_num, to_interpolate):
        try:
            msg = self._notifications[notif_num] % tuple(to_interpolate)
            self._log(msg)
        except:
            print('(_log_notification) error number:', notif_num)
            print('(_log_notification) to_interpolate:', to_interpolate)
            print('(_log_notification) self._notifications[notif_num]:', self._notifications[notif_num])
            raise

    def _get_issue_names_and_refs(self, issues, page_idx, url):
        res = list()
        ref_els = issues.xpath('div/p[@class="txt"]/a')
        self._log_notification(0, (page_idx, len(ref_els), url))
        for ref_el in ref_els:
            href = ''.join(ref_el.xpath('@href').extract())
            name = help.filter_text(''.join(ref_el.xpath('strong/text()').extract()), True, True)
            name = ' '.join(name.split()[:15])
            if len(name) == 0:
                self._log_error(0, (ref_el.extract(), page_idx, url))
                name = help.filter_text(''.join(ref_el.xpath('text()').extract()), True, True)
                if len(name) == 0:
                    self._log_error(1, (ref_el.extract(), page_idx, url))
                    name = 'unknown'
                else:
                    name = ' '.join(name.split()[:7])
            if len(href) == 0:
                self._log_error(2, (ref_el.extract(), page_idx, url))
            res.append((href, name))
        return res

    @staticmethod
    def _check_if_in_speakers(speaker_name, speakers_dict):
        if speaker_name in speakers_dict['map']:
            return True
        lst = speaker_name.split()
        perms = help.create_permutations(lst)
        for perm in perms:
            if tuple(perm) in speakers_dict['original_names']:
                return True
        return False

    def _add_speaker_abbreviations(self, speakers_dict, tpl, descriptor, issue_name, page_idx, url):
        if len(tpl) <= 2:
            speakers_dict['map'][tpl[0][0] + '. ' + tpl[1]] = descriptor
            speakers_dict['map'][(tpl[0][0] + '. ' + tpl[1]).upper()] = descriptor
        else:
            self._log_warning(7, (str(tpl), issue_name, page_idx, url))
            speakers_dict['map'][tpl[0][0] + '.' + tpl[1][0] + '.' + tpl[2]] = descriptor
            speakers_dict['map'][(tpl[0][0] + '.' + tpl[1][0] + '.' + tpl[2]).upper()] = descriptor
        return speakers_dict

    def _get_speakers(self, leader_sels, guest_sels, issue_name, page_idx, url, issue_folder):
        leaders = list()
        for leader_sel in leader_sels:
            name = leader_sel.xpath('b/text()').extract()
            if name is None:
                self._log_error(3, (issue_name, page_idx, url, leader_sel.extract()))
            name = help.fix_similar_looking_latin(help.filter_text(''.join(name), True, True))
            name = re.sub('[\n\.: ]+$', '', name)
            name = help.replace_yo(name)
            if len(name) == 0:
                self._log_error(4, (issue_name, page_idx, url))
            leaders.append(name)

        guests = list()
        for guest_sel in guest_sels:
            name = guest_sel.xpath('span[@class="about"]/strong/text()').extract()
            if name is None:
                self._log_error(5, (issue_name, page_idx, url, guest_sel.extract()))
            name = help.fix_similar_looking_latin(help.filter_text(''.join(name), True, True))
            name = re.sub('[\n\.: ]+$', '', name)
            name = help.replace_yo(name)
            if len(name) == 0:
                self._log_error(6, (issue_name, page_idx, url))
            guests.append(name)

        speakers_dict = help.construct(self._default_speakers)
        with open(issue_folder + '/default_speakers.txt', 'a') as f:
            for tpl, descr in speakers_dict['original_names'].items():
                if f.tell() > 0:
                    f.write('\n')
                f.write(descr + ': ' + ' '.join(tpl))

        for leader in leaders:
            if not self._check_if_in_speakers(leader, speakers_dict):
                self._log_notification(2, (leader, issue_name, page_idx, url))
                descriptor = str(len(speakers_dict['original_names'])) + 'в'
                with open(issue_folder + '/speakers.txt', 'a') as f:
                    if f.tell() > 0:
                        f.write('\n')
                    f.write(descriptor + ': ' + leader)
                tpl = tuple(leader.split())
                if len(tpl) > 0:
                    speakers_dict['original_names'][tpl] = descriptor
                    speakers_dict = self._add_speaker_abbreviations(
                        speakers_dict, tpl, descriptor, issue_name, page_idx, url)
            else:
                matches = help.get_matches(leader, speakers_dict['original_names'])
                if len(matches) == 0:
                    self._log_error(7, (leader, str(speakers_dict), issue_name, page_idx, url))
                elif len(matches) == 1:
                    descriptor = list(matches.values())[0]
                    with open(issue_folder + '/speakers.txt', 'a') as f:
                        if f.tell() > 0:
                            f.write('\n')
                        f.write(descriptor + ': ' + leader)
                elif len(matches) > 1:
                    descriptor = list(matches.values())[0]
                    self._log_error(9, (leader, str(speakers_dict), descriptor, issue_name, page_idx))
                    with open(issue_folder + '/speakers.txt', 'a') as f:
                        if f.tell() > 0:
                            f.write('\n')
                        f.write(descriptor + ': ' + leader)

        for guest in guests:
            if not self._check_if_in_speakers(guest, speakers_dict):
                self._log_notification(1, (guest, issue_name, page_idx, url))
                descriptor = str(len(speakers_dict['original_names']))
                with open(issue_folder + '/speakers.txt', 'a') as f:
                    if f.tell() > 0:
                        f.write('\n')
                    f.write(descriptor + ': ' + guest)
                tpl = tuple(guest.split())
                speakers_dict['original_names'][tpl] = descriptor
                speakers_dict = self._add_speaker_abbreviations(
                    speakers_dict, tpl, descriptor, issue_name, page_idx, url)
            else:
                matches = help.get_matches(guest, speakers_dict['original_names'])
                if len(matches) == 0:
                    self._log_error(8, (guest, str(speakers_dict), issue_name, page_idx, url))
                elif len(matches) == 1:
                    descriptor = list(matches.values())[0]
                    with open(issue_folder + '/speakers.txt', 'a') as f:
                        if f.tell() > 0:
                            f.write('\n')
                        f.write(descriptor + ': ' + guest)
                elif len(matches) > 1:
                    descriptor = list(matches.values())[0]
                    self._log_error(10, (guest, str(speakers_dict), descriptor, issue_name, page_idx, url))
                    with open(issue_folder + '/speakers.txt', 'a') as f:
                        if f.tell() > 0:
                            f.write('\n')
                        f.write(descriptor + ': ' + guest)
        return speakers_dict

    @staticmethod
    def _correct_format(string):
        # "А. Каменский"
        f1 = re.search('^([%s]\.*){1,2} *[%s][%s]+' % (help.uppercase_russian, help.uppercase_russian, help.lowercase_russian),
                       string)
        if f1 is None:
            return False
        return True

    def _check_if_speaker(self, node, std_inf):
        if help.get_elem_type(node) == 'b':
            t = node.xpath('text()').extract_first()
            if t is None:
                self._log_error(27, std_inf)
                return 'No'
            elif len(t) == 0:
                self._log_error(27, std_inf)
                return "No"
            if self._correct_format(t):
                return 'bold'
            self._log_error(28, help.unite_lists([node.extract()], std_inf))
            return 'No'
        else:
            if help.get_elem_type(node) == 'text':
                t = node.extract()
                m = re.match('[ \n\r\t]*(([%s]\.){1,2} *[%s]{2,})' %
                                    (help.uppercase_russian, help.uppercase_russian), t)
                if m is not None:
                    speaker_name = m.group(1)
                    # self._log_warning(6, help.unite_lists([speaker_name], std_inf))
                    return 'uppercase'
        return 'No'

    def _process_speaker_new_name(self, abbr, speakers, std_inf):
        if ':' in abbr:
            abbr = abbr.split(':')[0]
        matches = help.get_matches(abbr, speakers['original_names'])
        if len(matches) == 0:
            speaker_descr = str(len(speakers['original_names']))
            self._log_error(
                11, help.unite_lists(
                    [abbr, str(len(speakers['original_names'])), str(speakers['original_names'])],
                    std_inf))
            speakers['original_names'][tuple(abbr.split())] = speaker_descr
            speakers['map'][abbr] = speaker_descr
            speakers['map'][abbr.upper()] = speaker_descr
        elif len(matches) == 1:
            speakers['map'][abbr] = list(matches.values())[0]
            speaker_descr = list(matches.values())[0]
        elif len(matches) > 1:
            speakers['map'][abbr] = list(matches.values())[0]
            speaker_descr = list(matches.values())[0]
            self._log_error(
                12, help.unite_lists(
                    [len(matches), abbr, str(list(matches.keys())), str(list(speakers['original_names'].keys()))],
                    std_inf))
        return speaker_descr, speakers

    @staticmethod
    def _prepair_abbreviation(abbr):
        m = re.match('[ \n\r\t]*(([%s]\.){1,2} *[%s]{2,})' %
                     (help.uppercase_russian, help.uppercase_russian), abbr)
        if m is not None:
            abbr = m.group(1)
            abbr = re.sub('\. *', '. ', abbr)
        return abbr

    def _get_speaker_descriptor(self, speaker_name, speakers, std_inf):
        if speaker_name not in speakers['map']:
            speaker_descr, speakers = self._process_speaker_new_name(speaker_name,
                                                                     speakers,
                                                                     std_inf)
        else:
            speaker_descr = speakers['map'][speaker_name]
        return speaker_descr, speakers

    @staticmethod
    def _all_text(all_text_nodes):
        text = ''
        for node in all_text_nodes:
            if help.get_elem_type(node) == 'text':
                text += ''.join(node.extract())
            elif help.get_elem_type(node) == 'b':
                text += ''.join(node.xpath('text()').extract())
        return help.filter_text(text, True, True)

    @staticmethod
    def _strip_empty_start(all_text_nodes):
        idx = 0
        try:
            while idx < len(all_text_nodes) \
                    and help.get_elem_type(all_text_nodes[idx]) == 'text' \
                    and re.match('[ \n\r\t]*$', all_text_nodes[idx].extract()):
                idx += 1
        except TypeError:
            print("(_strip_empty_start)all_text_nodes:", all_text_nodes)
            print("(_strip_empty_start)all_text_nodes[idx]:", all_text_nodes[idx])
            print("(_strip_empty_start)all_text_nodes[idx].extract():", all_text_nodes[idx].extract())
            raise
        return all_text_nodes[idx:]

    def _check_if_several_replicas_in_paragraph(self, par, std_inf):
        several_replicas_in_paragraph = True
        bold_and_span = par.xpath('b | span')
        bolds = par.xpath('b')
        spans = par.xpath('span')
        if len(spans) > 1:
            self._log_warning(2, std_inf)
        types = help.get_types_list(bold_and_span)
        if len(spans) > len(bolds) == 0:
            self._log_warning(3, help.unite_lists([str(types)], std_inf))
            several_replicas_in_paragraph = False
        else:
            for el_idx, el in enumerate(bold_and_span):
                if help.get_elem_type(el) == 'b':
                    if el_idx == len(bold_and_span) - 1:
                        several_replicas_in_paragraph = False
                        self._log_warning(4, help.unite_lists([str(types)], std_inf))
                    else:
                        if help.get_elem_type(bold_and_span[el_idx + 1]) != 'span':
                            several_replicas_in_paragraph = False
                            self._log_warning(4, help.unite_lists([str(types)], std_inf))
        if several_replicas_in_paragraph:
            self._log_warning(5, help.unite_lists([str(types)], std_inf))
        return several_replicas_in_paragraph

    def _parse_paragraph(self, par, fd, speakers, current_speaker_descr, replica, repl_counter, std_inf):
        start = fd.tell()
        speaker_descr = current_speaker_descr
        new_speaker_descr = speaker_descr
        if help.get_elem_type(par) == 'p':
            all_text_nodes = par.xpath('b | text()')
        else:
            all_text_nodes = [par]
        bold = par.xpath('b')
        if len(all_text_nodes) == 0:
            self._log_error(13, help.unite_lists([par.extract()], std_inf))
        else:
            all_text_nodes = self._strip_empty_start(all_text_nodes)
            if len(all_text_nodes) == 0:
                pass
                # self._log_error(14, help.unite_lists([par.extract()], std_inf))
            else:
                if help.get_elem_type(all_text_nodes[0]) != 'b' and len(bold) > 0:
                    self._log_warning(
                        0, help.unite_lists(
                            [str(bold.extract()), help.get_elem_type(all_text_nodes[0]), all_text_nodes[0].extract()],
                            std_inf))
                for node in all_text_nodes:
                    cif = self._check_if_speaker(node, std_inf)
                    if cif != 'No':
                        if cif == 'bold':
                            speaker_name = ''.join(node.xpath('text()').extract())
                            t = ''
                        elif cif == 'uppercase':
                            # print('UPPERCASE')
                            t = node.extract()
                            t = re.sub('\\xa0', ' ', t)
                            m = re.match('[ \n\r\t]*(([%s]\.){1,2} *[%s]{2,})[\n–— \t\r-]*' %
                                         (help.uppercase_russian, help.uppercase_russian), t)
                            t = t[m.span()[1]:]
                            speaker_name = m.group(1)

                        speaker_name = help.replace_yo(speaker_name)
                        if len(speaker_name) == 0:
                            self._log_error(15, std_inf)
                        else:
                            speaker_name = ''.join(speaker_name)
                            if len(speaker_name) == 0:
                                self._log_error(15, std_inf)
                            else:
                                speaker_name = help.filter_text(speaker_name, True, True)
                                speaker_name = self._prepair_abbreviation(help.fix_similar_looking_latin(speaker_name))
                                speaker_name = re.sub('[\. :]+$', '', speaker_name)
                                new_speaker_descr, speakers = self._get_speaker_descriptor(
                                    speaker_name, speakers, std_inf)
                        if new_speaker_descr != speaker_descr:
                            repl_counter += 1
                            # print("(_parse_several_replicas_paragraph)replica:", replica)
                            replica = help.filter_text(replica, True, True)
                            fd.write(replica)
                            self._col_chars += len(replica)
                            replica = ''
                            if fd.tell() > 0:
                                fd.write('\n')
                                self._col_chars += 1
                            if fd.tell() == start:
                                self._log_error(29, std_inf)
                            start = fd.tell()
                            speaker_descr = new_speaker_descr
                            fd.write('<%s>' % speaker_descr)
                            self._col_chars += 2 + len(speaker_descr)
                        if re.search('НОВОСТИ', t) is None and re.search('РЕКЛАМА', t) is None:
                            replica += t
                    else:
                        # print("!!!!!!!!!adding")
                        if help.get_elem_type(node) == 'text':
                            t = node.extract()
                            if re.search('НОВОСТИ', t) is None and re.search('РЕКЛАМА', t) is None:
                                replica += t
                        else:
                            t = ''.join(node.xpath('text()').extract())
                            if re.search('НОВОСТИ', t) is None and re.search('РЕКЛАМА', t) is None:
                                replica += t
                        # print("(_parse_several_replicas_paragraph)replica:", replica)
        return speaker_descr, replica, repl_counter, speakers

    def _check_start(self, node, std_inf):
        text_start = (help.get_elem_type(node) == 'text') and self._check_if_speaker(node, std_inf)
        bold_start = (len(node.xpath('b')) != 0) and self._check_if_speaker(node.xpath('b')[0], std_inf)
        return text_start or bold_start

    def _parse_talk(self, talk_el, speakers, issue_name, page_idx, url, issue_folder):
        speaker_descr = '-1'
        repl_counter = 0
        pars = talk_el.xpath('p')
        text_els = talk_el.xpath('text()')
        if len(text_els) > 0:
            if len(pars) == 0:
                self._log_warning(8, (issue_name, page_idx, url))
                pars = talk_el.xpath('text()')
            else:
                start_idx = 0
                pars_and_text = talk_el.xpath('p | text()')
                while start_idx < len(pars) and not self._check_start(pars_and_text[start_idx],
                                                                      [issue_name, page_idx, start_idx, url]):
                    start_idx += 1
                pars = pars_and_text[start_idx:]
                # self._log_warning(9, (len(pars), len(text_els), issue_name, page_idx, url))
        else:
            if len(pars) == 0:
                self._log_error(19, (issue_name, page_idx, url))
            else:
                start_idx = 0
                while start_idx < len(pars) and len(pars[start_idx].xpath('b')) == 0:
                    start_idx += 1
                pars = pars[start_idx:]
        if len(pars) > 0:
            with open(issue_folder + '/transcript.txt', 'w') as f:
                replica = ''
                for par_idx, par in enumerate(pars):
                    std_inf = [issue_name, page_idx, par_idx, url]
                    speaker_descr, replica, repl_counter, speakers = self._parse_paragraph(
                        par, f, speakers, speaker_descr, replica, repl_counter, std_inf)
                f.write(replica + '\n')
                self._log_notification(3, [repl_counter, issue_name, page_idx, url])

    def start_requests(self):
        help.create_path(self._base_folder)
        self._save_logs_description()
        help.create_path(self._base_folder)
        urls = [self._start_url]
        for i in range(2, self._num_pages + 1):
            urls.append(self._start_url + 'archive/%s.html' % i)
        for page_idx, url in enumerate(urls):
            request = scrapy.Request(url=url, callback=self._page)
            request.meta['page_idx'] = page_idx
            yield request

    def _page(self, response):
        page_idx = response.meta['page_idx']
        issues = response.xpath('//body/div[@class="pagecontent"]/div[@class="main"]' +
                                '/div/section[@class="content"]/div/div[@id="archive"]/div[@class="rel"]/div')
        refs_and_names = self._get_issue_names_and_refs(issues, page_idx, response.url)
        for href, issue_name in refs_and_names:
            if len(href) > 0:
                url = response.urljoin(href)
                request = scrapy.Request(url, callback=self._issue_page)
                request.meta['page'] = page_idx
                request.meta['issue_name'] = issue_name
                yield request
            else:
                self._log_error(20, (issue_name, page_idx, response.url))

    def _issue_page(self, response):
        issue_name = response.meta['issue_name']
        page_idx = response.meta['page']
        url = response.url
        issue_folder = self._base_folder + issue_name
        issue_folder = help.add_index_to_filename_if_needed(issue_folder)
        help.create_path(issue_folder)
        relevant = response.xpath(
            'body[@class="mainpage"]/div[@class="pagecontent"]/div[@class="main"]/div/section[@class="content"]/div')
        if len(relevant) == 0:
            self._log_error(21, (issue_name, page_idx, url))
        speaker_block = relevant.xpath('div[@class="conthead discuss"]/div[contains(@class, "person pr")]')
        if len(speaker_block) == 0:
            self._log_error(22, (issue_name, page_idx, url))
        guests = speaker_block.xpath('div[@class="author iblock"]/a')
        leaders = speaker_block.xpath('div[contains(@class, "lead")]/a')

        if len(guests) == 0:
            self._log_error(23, (issue_name, page_idx, url))
        if len(leaders) == 0:
            self._log_error(24, (issue_name, page_idx, url))

        date_and_time = relevant.xpath('div[@class="conthead discuss"]/div[@class="titlebroadcastinfo clearfix"]' +
                                       '/div[@class="date left"]/strong')
        date_and_time = date_and_time.xpath('text()').extract_first()
        if date_and_time is None:
            self._log_error(25, (issue_name, page_idx, url))
        else:
            with open(issue_folder + '/release_inf.txt', 'w') as f:
                f.write(help.filter_text(date_and_time, True, True))
        speakers = self._get_speakers(leaders, guests, issue_name, page_idx, url, issue_folder)
        # print("(_issue_page)speakers['map']:", speakers['map'])
        talk_el = relevant.xpath('div[@class="multimedia mmtabs"]/div[@class="mmcontainer"]'
                                 '/div[@class="current mmread"]/div/div[contains(@class, "typical")]')
        if self._col_chars < self._total_char_limit:
            if len(talk_el) == 0:
                self._log_error(26, (issue_name, page_idx, url))
            else:
                self._parse_talk(talk_el, speakers, issue_name, page_idx, url, issue_folder)
     
         
        




