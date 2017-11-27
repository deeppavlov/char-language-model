import os
import re
import importlib.util
spec = importlib.util.spec_from_file_location("help", os.path.expanduser("~") +
                                              '/Natural-language-encoding/scrape/help_functions.py')
help = importlib.util.module_from_spec(spec)
spec.loader.exec_module(help)

import scrapy


class science_spider(scrapy.Spider):
    name = "science_spider"
    _base_folder = os.path.expanduser("~") + '/Natural-language-encoding/scrape/science2/science2_res/'
    _default_speakers = {'original_names': {('Кузичев', 'Анатолий'): '0в', ('Долгин', 'Борис'): '1в', ('Ицкович', 'Дмитрий'): '2в'},
                         'map': {'Кузичев': '0в', 'Долгин': '1в', 'Ицкович': '2в',
                                 'Кузичев Анатолий': '0в', 'Долгин Борис': '1в', 'Ицкович Дмитрий': '2в',
                                 'Анатолий Кузичев': '0в', 'Борис Долгин': '1в', 'Дмитрий Ицкович': '2в'}}

    def _log(self, msg):
        with open(self._base_folder + 'logs.txt', 'a') as f:
            f.write(msg + '\n') 

    def _get_issue_names_and_refs(self, issues, page_idx):
        res = list()
        ref_els = issues.xpath('div/div/h1/a')
        self._log('Number of links on page %s: %s' % (page_idx, len(ref_els)))
        for ref_el in ref_els:
            href = ''.join(ref_el.xpath('@href').extract())
            name = ''.join(ref_el.xpath('text()').extract())
            name = ' '.join(name.split()[:15])
            if len(name) == 0:
                self._log('Error: no issue name in reference element ---%s--- on page %s' % (ref_el.extract_first(), page_idx))
            if len(href) == 0:
                self._log('Error: no link in reference element ---%s--- on page %s' % (ref_el.extract_first(), page_idx))
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
    
    def _get_speakers(self, sels, issue_name, page_idx, issue_folder):
        speakers = list()
        for sel in sels:
            name = sel.xpath('a/div[@class="presenter__name"]/text()').extract_first()
            position = sel.xpath('a/div[@class="presenter__position"]/text()').extract_first()
            if name is None:
                self._log("Error: failed to get speaker name on issue \'%s\' on page %s. Selector: %s" %
                          (issue_name, page_idx, sel.extract()))
            if position is None:
                self._log("Error: failed to get speaker position on issue \'%s\' on page %s. Selector: %s" %
                          (issue_name, page_idx, sel.extract()))
            name = help.fix_similar_looking_latin(help.filter_text(''.join(name), True, True))
            name = re.sub('[\n\.: ]+$', '', name)
            position = help.fix_similar_looking_latin(help.filter_text(''.join(position), True, True))
            speakers.append((name, position))
        speakers_dict = help.construct(self._default_speakers)
        with open(issue_folder + '/default_speakers.txt', 'w') as f:
            for tpl, descr in speakers_dict['original_names'].items():
                if f.tell() > 0:
                    f.write('\n')
                f.write(descr + ': ' + ' '.join(tpl))
        for speaker in speakers:
            if not self._check_if_in_speakers(speaker[0], speakers_dict):
                descriptor = str(len(speakers_dict['original_names']))
                if re.search('[Вв]едущий', speaker[1]) is not None:
                    descriptor += 'в'
                with open(issue_folder + '/speakers.txt', 'w') as f:
                    if f.tell() > 0:
                        f.write('\n')
                    f.write(descriptor + ': ' + speaker[0])
                tpl = tuple(speaker[0].split())
                speakers_dict['original_names'][tpl] = descriptor
                speakers_dict['map'][tpl[1]] = descriptor
        return speakers_dict

    @staticmethod
    def _find_following_text(sel_idx, part_sels):
        # print('(_find_following_text)part_sels:', part_sels)
        text = ''
        idx = sel_idx + 1
        try:
            while len(text) == 0 and idx < len(part_sels):
                if help.get_elem_type(part_sels[idx]) == 'strong':
                    text += ''.join(part_sels[idx].xpath('text()').extract())
                else:
                    text += ''.join(part_sels[idx].extract())
                idx += 1
        except TypeError:
            print('(_find_following_text)part_sels:', part_sels)
            raise
        return text

    @staticmethod
    def _correct_format(string):
        # "Иванов" или "Иванов "
        f1 = re.search('^[%s][%s-]+ ?$' % (help.uppercase_russian, help.lowercase_russian), string)
        # "И.И.:" или "И.И:"
        f2 = re.search('^[%s]\.* *[%s]\.*:* ?' % (help.uppercase_russian, help.uppercase_russian), string)
        # "Иванов:" или "Иванов: " или "Иванов." или "Иванов. "
        f3 = re.search('^[%s][%s]+[:\.] *$' % (help.uppercase_russian, help.lowercase_russian), string)
        # "ИВАНОВ ИВАН: "
        f4 = re.search('^[%s]{2,} [%s]{2,}: ?$' % (help.uppercase_russian, help.uppercase_russian), string)
        # "Иванов Иван:" или "Иванов Иван." или "Иванов Иван: " или "Иванов Иван. "
        f5 = re.search('^[%s][%s]+ [%s][%s]+[:\.]? ?' % (help.uppercase_russian, help.lowercase_russian,
                                                         help.uppercase_russian, help.lowercase_russian),
                       string)
        # "ИИ:"
        f6 = re.search('^[%s][%s]:$' % (help.uppercase_russian, help.uppercase_russian), string)
        # "Ел.Б-О.:" (Елизавета Бонч-Осмоловская)
        f7 = re.search('^[%s][%s]\.[%s]-[%s]\.:' % (help.uppercase_russian, help.lowercase_russian,
                                                    help.uppercase_russian, help.uppercase_russian),
                       string)
        # "А.Каменский"
        f8 = re.search('^[%s]\.[%s][%s]+' % (help.uppercase_russian, help.uppercase_russian, help.lowercase_russian),
                       string)
        if f1 is None and f2 is None and f3 is None and f4 is None and \
                f5 is None and f6 is None and f7 is None and f8 is None:
            return False
        return True

    def _check_if_speaker(self, sel, sel_idx, part_sels, issue_name, page_idx, par_idx):
        t = sel.xpath('text()').extract_first()
        if help.get_elem_type(sel) == 'strong':
            if t is None:
                self._log(("Warning: there is strong element but it does not contain text." +
                           " Issue: '%s', page: %s, paragraph number: %s, sel_idx: %s") %
                          (issue_name, page_idx, par_idx, sel_idx))
                return False
            elif len(t) == 0:
                self._log(("Warning: there is strong element but it contains zero length text." +
                           " Issue: '%s', page: %s, paragraph number: %s, sel_idx: %s") %
                          (issue_name, page_idx, par_idx, sel_idx))
                return False
            fol_text = self._find_following_text(sel_idx, part_sels)
            if self._correct_format(t):
                return True
            self._log(('Warning: \'%s\' is strong element but not speaker name.' +
                       ' Issue: \'%s\', page: %s, paragraph number: %s, sel_idx: %s') % (t, issue_name, page_idx,
                                                                                         par_idx, sel_idx))
            return False
        return False

    def _process_speaker_new_name(self, abbr, speakers, issue_name, page_idx, par_idx, sel_idx):
        matches = help.get_matches(abbr, speakers['original_names'])
        if len(matches) == 0:
            speaker_descr = str(len(speakers['original_names']))
            self._log("Error: no matches found for abbreviation '%s', falling to '3' speakers: '%s'" %
                      (abbr, str(speakers['original_names'])) +
                      "\n\tissue_name: %s, page_idx: %s, par_idx: %s, sel_idx: %s" % (issue_name, page_idx,
                                                                                      par_idx, sel_idx))
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
                       "\n\tissue_name: %s, page_idx: %s, par_idx: %s, sel_idx: %s" % (issue_name, page_idx,
                                                                                       par_idx, sel_idx))
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
            
    def _parse_paragraph(self, par, fd, speakers, issue_name, page_idx, par_idx):
        fd_start = fd.tell()
        speakers = help.construct(speakers)
        part_sels = par.xpath('strong | b/text() | text() | i/text() | blockquote/text() | nobr/text()')
        replica = ''
        for sel_idx, sel in enumerate(part_sels):
            if self._check_if_speaker(sel, sel_idx, part_sels, issue_name, page_idx, par_idx):

                speaker_name = help.filter_text(
                                   help.fix_similar_looking_latin(
                                       ''.join(sel.xpath('text()').extract())),
                                   True,
                                   True)
                speaker_name = self._prepair_abbreviation(speaker_name)
                speaker_name = re.sub('[\. :]+$', '', speaker_name)
                if speaker_name not in speakers['map']:
                    speaker_descr, speakers = self._process_speaker_new_name(speaker_name,
                                                                             speakers,
                                                                             issue_name,
                                                                             page_idx,
                                                                             par_idx,
                                                                             sel_idx)
                else:
                    speaker_descr = speakers['map'][speaker_name]
                if len(replica) > 0:
                    # print("filtering '%s'" % issue_name)
                    replica = help.filter_text(replica, True, True)
                    fd.write(replica)
                elif fd.tell() > fd_start:
                    self._log("Error: empty replica. speaker_descr." +
                              "\n\tissue_name: %s, page_idx: %s, par_idx: %s, sel_idx: %s" % (issue_name, page_idx,
                                                                                              par_idx, sel_idx))
                replica = ''
                if fd.tell() > 0:
                    fd.write('\n')
                fd.write('<%s>' % speaker_descr)
            else:
                if help.get_elem_type(sel) == 'strong':
                    t = ''.join(sel.xpath('text()').extract())
                else:
                    t = sel.extract()
                replica += t
        if len(replica) > 0:
            # print("filtering '%s'" % issue_name)
            replica = help.filter_text(replica, True, True)
            fd.write(replica)
        elif fd.tell() > fd_start:
            self._log("Error: empty replica. speaker_descr." +
                      "\n\tissue_name: %s, page_idx: %s, par_idx: %s, sel_idx: %s" % (issue_name, page_idx,
                                                                                      par_idx, sel_idx))
        return speakers

    @staticmethod
    def _retrieve_introduction(intr_el):
        strings = intr_el.xpath('text() | a/text() | b/text() | i/text() | blockquote/text()').extract()
        return help.filter_text(''.join(strings), True, True)

    def _parse_talk(self, talk_el, speakers, issue_name, page_idx, issue_folder):
        introduction_el = talk_el.xpath('p[1]/em')
        if len(introduction_el) == 0:
            self._log("Error: No introduction found in issue \'%s\' on page %s" % (issue_name, page_idx))
        else:
            intr = self._retrieve_introduction(introduction_el)
            with open(issue_folder + '/introduction.txt', 'w') as f:
                f.write(intr)
        pars = talk_el.xpath('p[strong]')
        if len(pars) == 0:
            self._log("Error: No paragraphs found in issue '%s' on page %s" % (issue_name, page_idx))
        with open(issue_folder + '/transcript.txt', 'w') as f:
            for par_idx, par in enumerate(pars):
                speakers = self._parse_paragraph(par, f, speakers, issue_name, page_idx, par_idx)

    def start_requests(self):
        help.create_path(self._base_folder)
        urls = ['https://radiovesti.ru/brand/60941/page/%s/' % i for i in range(1, 10)]
        for page_idx, url in enumerate(urls):
            request = scrapy.Request(url=url, callback=self._page)
            request.meta['page_idx'] = page_idx + 1
            yield request

    def _page(self, response):
        page_idx = response.meta['page_idx']
        issues = response.xpath('//body/div/div/div/div/div/div' +
                                '[@class="programms-page__list-wrap js-list-wrap  view-block"]' +
                                '/div[not(@class="news banner") and not(@class="clearfix")]')
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
        issue_folder = help.add_index_to_filename_if_needed(issue_folder)
        help.create_path(issue_folder)
        relevant = response.xpath('//body/div/div[@class="main"]/div/div[@class="main__content"]/div')
        if len(relevant) == 0:
            self._log('Error: no relevant element on issue \'%s\' on page %s' % (issue_name, page_idx))
        speakers = relevant.xpath('div[@class="inside-page__header"]' +
                                  '/div[@class="news__content"]/div[@class="page__top-presenter"]/div')
        if len(speakers) == 0:
            self._log('Error: no speaker elements found on issue \'%s\' on page %s' % (issue_name, page_idx))

        date_and_time = relevant.xpath('div[@class="inside-page__header"]' +
                                       '/div[@class="news__content"]/div[@class="news__date"]')
        date_and_time = date_and_time.xpath('text()').extract_first()
        if date_and_time is None:
            self._log('Error: No date and time were found. Page: %s, issue name: \'%s\'' % (page_idx, issue_name))
        else:
            with open(issue_folder + '/release_inf.txt', 'w') as f:
                f.write(help.filter_text(date_and_time, True, True))
        speakers = self._get_speakers(speakers, issue_name, page_idx, issue_folder)
        talk_el = relevant.xpath('div[@class="insides-page__news clearfix"]' +
                                 '/div[@class="insides-page__news__text  insides-page__news__text_with-popular"]')
        if len(talk_el) == 0:
            self._log('Error: no talk element on issue \'%s\' on page %s' % (issue_name, page_idx)) 
        else:
            self._parse_talk(talk_el, speakers, issue_name, page_idx, issue_folder)
     
         
        




