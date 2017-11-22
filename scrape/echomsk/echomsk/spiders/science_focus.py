import os
import re
import importlib.util
spec = importlib.util.spec_from_file_location("help", os.path.expanduser("~") +
                                              '/Natural-language-encoding/scrape/help_functions.py')
help = importlib.util.module_from_spec(spec)
spec.loader.exec_module(help)

import scrapy


class science_focus(scrapy.Spider):
    name = "science_focus"
    _base_folder = os.path.expanduser("~") + '/Natural-language-encoding/scrape/echomsk/science_focus_res/'
    _default_speakers = {'original_names': {('Асадова', 'Наргиз'): '0в', ('Быковский', 'Егор'): '1в'},
                         'map': {'Н. Асадова': '0в', 'Е. Быковский': '1в'}}

    def _log(self, msg):
        with open(self._base_folder + 'logs.txt', 'a') as f:
            f.write(msg + '\n') 

    def _get_issue_names_and_refs(self, issues, page_idx):
        res = list()
        ref_els = issues.xpath('div/p[@class="txt"]/a')
        self._log('Number of links on page %s: %s' % (page_idx, len(ref_els)))
        for ref_el in ref_els:
            href = ''.join(ref_el.xpath('@href').extract())
            name = help.filter_text(''.join(ref_el.xpath('strong/text()').extract()), True, True)
            name = ' '.join(name.split()[:15])
            if len(name) == 0:
                self._log('Error: no issue name in reference element ---%s--- on page %s' %
                          (ref_el.extract(), page_idx))
                name = help.filter_text(''.join(ref_el.xpath('text()').extract()), True, True)
                if len(name) == 0:
                    self._log("Error: totally failed to get name for issue in reference element"
                              " ---%s--- on page %s. Falling to unknown" % (ref_el.extract(), page_idx))
                    name = 'unknown'
                else:
                    name = ' '.join(name.split()[:7])
            if len(href) == 0:
                self._log('Error: no link in reference element ---%s--- on page %s' % (ref_el.extract(), page_idx))
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
    
    def _get_speakers(self, leader_sels, guest_sels, issue_name, page_idx, issue_folder):
        leaders = list()
        for leader_sel in leader_sels:
            name = leader_sel.xpath('b/text()').extract()
            if name is None:
                self._log("Error: failed to get leader name on issue \'%s\' on page %s. Selector: %s" %
                          (issue_name, page_idx, leader_sel.extract()))
            name = help.fix_similar_looking_latin(help.filter_text(''.join(name), True, True))
            name = re.sub('[\n\.: ]+$', '', name)
            if len(name) == 0:
                self._log('Error: zero length name of leader.\n\tIssue: \'%s\', page: \'%s\'' % (issue_name, page_idx))
            leaders.append(name)

        guests = list()
        for guest_sel in guest_sels:
            name = guest_sel.xpath('span[@class="about"]/strong/text()').extract()
            if name is None:
                self._log("Error: failed to get leader name on issue \'%s\' on page %s. Selector: %s" %
                          (issue_name, page_idx, guest_sel.extract()))
            name = help.fix_similar_looking_latin(help.filter_text(''.join(name), True, True))
            name = re.sub('[\n\.: ]+$', '', name)
            if len(name) == 0:
                self._log('Error: zero length name of guest.\n\tIssue: \'%s\', page: \'%s\'' % (issue_name, page_idx))
            guests.append(name)

        speakers_dict = help.construct(self._default_speakers)
        with open(issue_folder + '/default_speakers.txt', 'w') as f:
            for tpl, descr in speakers_dict['original_names'].items():
                if f.tell() > 0:
                    f.write('\n')
                f.write(descr + ': ' + ' '.join(tpl))

        for leader in leaders:
            if not self._check_if_in_speakers(leader, speakers_dict):
                self._log('Notification: %s leader is added.\n\tIssue: \'%s\', page: %s' %
                          (leader, issue_name, page_idx))
                descriptor = str(len(speakers_dict['original_names'])) + 'в'
                with open(issue_folder + '/speakers.txt', 'w') as f:
                    if f.tell() > 0:
                        f.write('\n')
                    f.write(descriptor + ': ' + leader)
                tpl = tuple(leader.split())
                speakers_dict['original_names'][tpl] = descriptor
                speakers_dict['map'][tpl[0][0] + '. ' + tpl[1]] = descriptor
            else:
                matches = help.get_matches(leader, speakers_dict['original_names'])
                if len(matches) == 0:
                    self._log("Error: leader name '%s' is found in speakers dictionary ---%s--- by"
                              " _check_if_in_speakers method but not found by get_matches functions."
                              "\n\tIssue: '%s', page: %s" % (leader, str(speakers_dict), issue_name, page_idx))
                elif len(matches) == 1:
                    descriptor = list(matches.values())[0]
                    with open(issue_folder + '/speakers.txt', 'w') as f:
                        if f.tell() > 0:
                            f.write('\n')
                        f.write(descriptor + ': ' + leader)
                elif len(matches) > 1:
                    descriptor = list(matches.values())[0]
                    self._log("Error: more than 1 match found for leader '%s' in speakers dictionary (%s)."
                              " Falling to '%s'.\n\tIssue: '%s', page: %s" %
                              (leader, str(speakers_dict), descriptor, issue_name, page_idx))
                    with open(issue_folder + '/speakers.txt', 'w') as f:
                        if f.tell() > 0:
                            f.write('\n')
                        f.write(descriptor + ': ' + leader)

        for guest in guests:
            if not self._check_if_in_speakers(guest, speakers_dict):
                self._log('Notification: %s guest is added.\n\tIssue: \'%s\', page: %s' %
                          (guest, issue_name, page_idx))
                descriptor = str(len(speakers_dict['original_names']))
                with open(issue_folder + '/speakers.txt', 'w') as f:
                    if f.tell() > 0:
                        f.write('\n')
                    f.write(descriptor + ': ' + guest)
                tpl = tuple(guest.split())
                speakers_dict['original_names'][tpl] = descriptor
                speakers_dict['map'][tpl[0][0] + '. ' + tpl[1]] = descriptor
            else:
                matches = help.get_matches(guest, speakers_dict['original_names'])
                if len(matches) == 0:
                    self._log("Error: guest name '%s' is found in speakers dictionary ---%s--- by"
                              " _check_if_in_speakers method but not found by get_matches functions."
                              "\n\tIssue: '%s', page: %s" % (guest, str(speakers_dict), issue_name, page_idx))
                elif len(matches) == 1:
                    descriptor = list(matches.values())[0]
                    with open(issue_folder + '/speakers.txt', 'w') as f:
                        if f.tell() > 0:
                            f.write('\n')
                        f.write(descriptor + ': ' + guest)
                elif len(matches) > 1:
                    descriptor = list(matches.values())[0]
                    self._log("Error: more than 1 match found for guest '%s' in speakers dictionary (%s)."
                              " Falling to '%s'.\n\tIssue: '%s', page: %s" %
                              (guest, str(speakers_dict), descriptor, issue_name, page_idx))
                    with open(issue_folder + '/speakers.txt', 'w') as f:
                        if f.tell() > 0:
                            f.write('\n')
                        f.write(descriptor + ': ' + guest)
        return speakers_dict

    # @staticmethod
    # def _correct_format(string):
    #     # "Иванов" или "Иванов "
    #     f1 = re.search('^[%s][%s-]+ ?$' % (help.uppercase_russian, help.lowercase_russian), string)
    #     # "И.И.:" или "И.И:"
    #     f2 = re.search('^[%s]\.* *[%s]\.*:* ?' % (help.uppercase_russian, help.uppercase_russian), string)
    #     # "Иванов:" или "Иванов: " или "Иванов." или "Иванов. "
    #     f3 = re.search('^[%s][%s]+[:\.] *$' % (help.uppercase_russian, help.lowercase_russian), string)
    #     # "ИВАНОВ ИВАН: "
    #     f4 = re.search('^[%s]{2,} [%s]{2,}: ?$' % (help.uppercase_russian, help.uppercase_russian), string)
    #     # "Иванов Иван:" или "Иванов Иван." или "Иванов Иван: " или "Иванов Иван. "
    #     f5 = re.search('^[%s][%s]+ [%s][%s]+[:\.]? ?' % (help.uppercase_russian, help.lowercase_russian,
    #                                                      help.uppercase_russian, help.lowercase_russian),
    #                    string)
    #     # "ИИ:"
    #     f6 = re.search('^[%s][%s]:$' % (help.uppercase_russian, help.uppercase_russian), string)
    #     # "Ел.Б-О.:" (Елизавета Бонч-Осмоловская)
    #     f7 = re.search('^[%s][%s]\.[%s]-[%s]\.:' % (help.uppercase_russian, help.lowercase_russian,
    #                                                 help.uppercase_russian, help.uppercase_russian),
    #                    string)
    #     # "А.Каменский"
    #     f8 = re.search('^[%s]\.[%s][%s]+' % (help.uppercase_russian, help.uppercase_russian, help.lowercase_russian),
    #                    string)
    #     if f1 is None and f2 is None and f3 is None and f4 is None and \
    #             f5 is None and f6 is None and f7 is None and f8 is None:
    #         return False
    #     return True
    #
    # def _check_if_speaker(self, sel, sel_idx, part_sels, issue_name, page_idx, par_idx):
    #     t = sel.xpath('text()').extract_first()
    #     if help.get_elem_type(sel) == 'strong':
    #         if t is None:
    #             self._log(("Warning: there is strong element but it does not contain text." +
    #                        " Issue: '%s', page: %s, paragraph number: %s, sel_idx: %s") %
    #                       (issue_name, page_idx, par_idx, sel_idx))
    #             return False
    #         elif len(t) == 0:
    #             self._log(("Warning: there is strong element but it contains zero length text." +
    #                        " Issue: '%s', page: %s, paragraph number: %s, sel_idx: %s") %
    #                       (issue_name, page_idx, par_idx, sel_idx))
    #             return False
    #         fol_text = self._find_following_text(sel_idx, part_sels)
    #         if self._correct_format(t):
    #             return True
    #         self._log(('Warning: \'%s\' is strong element but not speaker name.' +
    #                    ' Issue: \'%s\', page: %s, paragraph number: %s, sel_idx: %s') % (t, issue_name, page_idx,
    #                                                                                      par_idx, sel_idx))
    #         return False
    #     return False

    def _process_speaker_new_name(self, abbr, speakers, issue_name, page_idx, par_idx):
        matches = help.get_matches(abbr, speakers['original_names'])
        if len(matches) == 0:
            speaker_descr = str(len(speakers['original_names']))
            self._log("Error: no matches found for abbreviation '%s', falling to '%s' speakers: '%s'" %
                      (abbr, str(len(speakers['original_names'])), str(speakers['original_names'])) +
                      "\n\tissue_name: %s, page_idx: %s, par_idx: %s" % (issue_name, page_idx, par_idx))
            speakers['original_names'][tuple(abbr.split())] = speaker_descr
            speakers['map'][abbr] = speaker_descr
        elif len(matches) == 1:
            speakers['map'][abbr] = list(matches.values())[0]
            speaker_descr = list(matches.values())[0]
        elif len(matches) > 1:
            speakers['map'][abbr] = list(matches.values())[0]
            speaker_descr = list(matches.values())[0]
            self._log("Error: %s matches for abbreviation '%s' found. Found abbreviations: %s. Speakers: %s" % 
                      (len(matches),
                       abbr,
                       str(list(matches.keys())),
                       str(list(speakers['original_names'].keys()))) +
                       "\n\tissue_name: %s, page_idx: %s, par_idx: %s" % (issue_name, page_idx, par_idx))
        return speaker_descr, speakers

    @staticmethod
    def _prepair_abbreviation(abbr):
        f3 = re.search('^[%s][%s]+[:\.] *$' % (help.uppercase_russian, help.lowercase_russian), abbr)
        # "ИИ:"
        f6 = re.search('^[%s][%s]:$' % (help.uppercase_russian, help.uppercase_russian), abbr)
        # "Ел.Б-О.:" (Елизавета Бонч-Осмоловская)
        f7 = re.search('^[%s][%s]\.[%s]-[%s]\.:' % (help.uppercase_russian, help.lowercase_russian,
                                                    help.uppercase_russian, help.uppercase_russian),
                       abbr)
        if f6 is not None:
            abbr = ' '.join(list(re.sub('[^%s]' % help.uppercase_russian, '', abbr)))
        elif f3 is not None:
            abbr = ' '.join([
                s.title() for s in re.sub(
                            '[^%s%s]' % (help.uppercase_russian, help.lowercase_russian), '', abbr).split()])
        elif f7 is not None:
            aspl = abbr.split('-')
            abbr = ' '.join(aspl.split('.'))
        return abbr

    def _get_speaker_descriptor(self, speaker_name, speakers, issue_name, page_idx, par_idx):
        if speaker_name not in speakers['map']:
            speaker_descr, speakers = self._process_speaker_new_name(speaker_name,
                                                                     speakers,
                                                                     issue_name,
                                                                     page_idx,
                                                                     par_idx)
        else:
            speaker_descr = speakers['map'][speaker_name]
        return speaker_descr, speakers

    def _check_nodes(self, par, issue_name, page_idx, par_idx):
        good_nodes = ['b', 'span']
        all_nodes = par.xpath('child::*')
        for node in all_nodes:
            n_type = help.get_elem_type(node)
            if n_type not in good_nodes:
                self._log("Error: bad '%s' node found"
                          "\n\tissue_name: %s, page_idx: %s, par_idx: %s" % (n_type, issue_name,
                                                                             page_idx, par_idx))

    @staticmethod
    def _all_text(all_text_nodes):
        text = ''
        for node in all_text_nodes:
            if help.get_elem_type(node) == 'text':
                text += ''.join(node.extract())
            elif help.get_elem_type(node) == 'b':
                text += ''.join(node.xpath('text()').extract())
        return help.filter_text(text, True, True)
            
    def _parse_paragraph(self, par, fd, speakers, current_speaker_descr, issue_name, page_idx, par_idx):
        speakers = help.construct(speakers)
        bold = par.xpath('b')
        bold_and_span = par.xpath('b | span')
        all_text_nodes = par.xpath('b | text()')
        std_inf = "\n\tissue_name: %s, page: %s, par_idx: %s" % (issue_name, page_idx, par_idx)



        if len(all_text_nodes) == 0:
            self._log("Error: paragraph is totally empty" + std_inf)
        else:
            if help.get_elem_type(all_text_nodes[0]) != 'b' and len(bold) > 0:
                self._log("Warning: bold element is not first."
                          "\n\tbold_elements: %s, first element: %s"
                          % (str(bold.extract()), help.get_elem_type(all_text_nodes[0])) + std_inf)
            if len(bold) > 1:
                self._log("Warning: more than one bold element.\n\tbold_elements: %s" % str(bold.extract()) + std_inf)
            if len(bold) > 0:
                if help.get_elem_type(bold_and_span[0]) == 'b' and help.get_elem_type(bold_and_span[1]) == 'span':
                    speaker_name = bold_and_span[0].extract()
                    if len(speaker_name) == 0:
                        self._log("Error: speaker syntax is provider but no speaker" + std_inf)
                        speaker_descr = current_speaker_descr
                    else:
                        speaker_name = ''.join(speaker_name)
                        if len(speaker_name) == 0:
                            self._log("Error: speaker syntax is provider but no speaker" + std_inf)
                            speaker_descr = current_speaker_descr
                        else:
                            speaker_name = help.filter_text(speaker_name, True, True)
                            speaker_name = self._prepair_abbreviation(help.fix_similar_looking_latin(speaker_name))
                            speaker_name = re.sub('[\. :]+$', '', speaker_name)
                            speaker_descr, speakers = self._get_speaker_descriptor(
                                speaker_name, speakers, issue_name, page_idx, par_idx)
                else:
                    speaker_descr = current_speaker_descr
            else:
                speaker_descr = current_speaker_descr
            if help.get_elem_type(all_text_nodes[0]) == 'b':
                all_text_nodes = all_text_nodes[1:]
                if len(all_text_nodes) == 0:
                    self._log("Error: paragraph does not contain text nodes while speaker is provided" + std_inf)
            text = self._all_text(all_text_nodes)
            if len(text) == 0:
                self._log("Error: no text.\n\tbold_elements: %s" % str(bold.extract()) + std_inf)
            if speaker_descr != current_speaker_descr:
                if fd.tell() > 0:
                    fd.write('\n')
                fd.write('<%s>' % speaker_descr + text)
            else:
                if speaker_descr == '-1':
                    self._log("Error: speaker descriptor is not provided." + std_inf)
                fd.write(text)
        return speaker_descr, speakers

    def _parse_talk(self, talk_el, speakers, issue_name, page_idx, issue_folder):
        speaker_descr = '-1'
        pars = talk_el.xpath('p')
        if len(pars) == 0:
            self._log("Error: No paragraphs found in issue '%s' on page %s" % (issue_name, page_idx))
        else:
            with open(issue_folder + '/transcript.txt', 'w') as f:
                for par_idx, par in enumerate(pars):
                    speaker_descr, speakers = self._parse_paragraph(par, f, speakers, speaker_descr,
                                                                    issue_name, page_idx, par_idx)

    def start_requests(self):
        help.create_path(self._base_folder)
        urls = ['https://echo.msk.ru/programs/naukafokus/']
        for i in range(2, 7):
            urls.append('https://echo.msk.ru/programs/naukafokus/archive/%s.html' % i)
        for page_idx, url in enumerate(urls):
            request = scrapy.Request(url=url, callback=self._page)
            request.meta['page_idx'] = page_idx
            yield request

    def _page(self, response):
        page_idx = response.meta['page_idx']
        issues = response.xpath('//body/div[@class="pagecontent"]/div[@class="main"]' +
                                '/div/section[@class="content"]/div/div[@id="archive"]/div[@class="rel"]/div')
        refs_and_names = self._get_issue_names_and_refs(issues, page_idx)
        for href, issue_name in refs_and_names:
            if len(href) > 0:
                url = response.urljoin(href)
                request = scrapy.Request(url, callback=self._issue_page)
                request.meta['page'] = page_idx
                request.meta['issue_name'] = issue_name
                yield request
            else:
                self._log("Error: zero length reference for issue '%s', page: %s'" % (issue_name, page_idx)) 

    def _issue_page(self, response):
        issue_name = response.meta['issue_name']
        page_idx = response.meta['page']
        issue_folder = self._base_folder + issue_name
        help.create_path(issue_folder)
        relevant = response.xpath(
            'body[@class="mainpage"]/div[@class="pagecontent"]/div[@class="main"]/div/section[@class="content"]/div')
        if len(relevant) == 0:
            self._log('Error: no relevant element on issue \'%s\' on page %s' % (issue_name, page_idx))
        speaker_block = relevant.xpath('div[@class="conthead discuss"]/div[@class="person pr2"]')
        if len(speaker_block) == 0:
            self._log('Error: no speaker block found on issue \'%s\' on page %s' % (issue_name, page_idx))
        guests = speaker_block.xpath('div[@class="author iblock"]/a')
        leaders = speaker_block.xpath('div[contains(@class, "lead")]/a')

        if len(guests) == 0:
            self._log('Error: no guest elements found on issue \'%s\' on page %s' % (issue_name, page_idx))
        if len(leaders) == 0:
            self._log('Error: no leader elements found on issue \'%s\' on page %s' % (issue_name, page_idx))

        date_and_time = relevant.xpath('div[@class="conthead discuss"]/div[@class="titlebroadcastinfo clearfix"]' +
                                       '/div[@class="date left"]/strong')
        date_and_time = date_and_time.xpath('text()').extract_first()
        if date_and_time is None:
            self._log('Error: No date and time were found. Page: %s, issue name: \'%s\'' % (page_idx, issue_name))
        else:
            with open(issue_folder + '/release_inf.txt', 'w') as f:
                f.write(help.filter_text(date_and_time, True, True))
        speakers = self._get_speakers(leaders, guests, issue_name, page_idx, issue_folder)
        talk_el = relevant.xpath('div[@class="multimedia mmtabs"]/div[@class="mmcontainer"]'
                                 '/div[@class="current mmread"]/div/div[contains(@class, "typical dialog")]')
        if len(talk_el) == 0:
            self._log('Error: no talk element on issue \'%s\' on page %s' % (issue_name, page_idx)) 
        else:
            self._parse_talk(talk_el, speakers, issue_name, page_idx, issue_folder)
     
         
        




