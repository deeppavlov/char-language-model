import scrapy
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



class GordonBold(scrapy.Spider):
    name = "gordon_bold"
    _all_file_name = os.path.expanduser("~") + '/gordon/gordon_bold/all.txt'

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
                        yield scrapy.Request(url, callback=self._issue_page)

    def _issue_page(self, response):
        issue_name_list = response.css('body>table>tr>td>h4::text').extract()
        if len(issue_name_list) == 0:
            issue_name = 'unknown'
        else:
            if len(issue_name_list[0]) == 0:
                issue_name = 'unknown'
            else:
                issue_name = issue_name_list[0] 
        issue_file_name = add_index_to_filename_if_needed(os.path.expanduser("~") + '/gordon/gordon_bold/' + issue_name + '.txt')
        create_path(issue_file_name, file_name_is_in_path=True)
        with open(issue_file_name, 'w') as issue_file:
            with open(self._all_file_name, 'a') as all_file:
                td = response.xpath('//body/table/tr/td[a[text()="Стенограмма эфира"]]')
                bold_strings = td.xpath('//b/text()').extract()
                for string in bold_strings:
                    if string != '\n':
                        issue_file.write('\n' + '-'*20 + '\n')
                        issue_file.write(string)
                        all_file.write('\n' + '-'*20 + '\n')
                        all_file.write(string)

