$/=">";                     # input record separator
open(input, "<$ARGV[0]");
open(output, ">$ARGV[1]");
$lowercase ='abcdefghijklmnopqrstuvwxyz';
$uppercase ='ABCDEFGHIJKLMNOPQRSTUVWXYZ';
$other = " \\n\\t\\.,\\(\\)\\?\\!\"'-";
while (<input>) {
  if (/<text /) {$text=1;}  # remove all but between <text> ... </text>
  if (/#redirect/i) {$text=0;}  # remove #REDIRECT
  if ($text) {

    # Remove any text not normally visible
    if (/<\/text>/) {$text=0;}
    s/<.*>//;               # remove xml tags
    s/&amp;/&/g;            # decode URL encoded chars
    s/&lt;/</g;
    s/&gt;/>/g;
    s/<ref[^<]*<\/ref>//g;  # remove references <ref...> ... </ref>
    s/<[^>]*>//g;           # remove xhtml tags
    s/\[http:[^] ]*/[/g;    # remove normal url, preserve visible text
    s/\|thumb//ig;          # remove images links, preserve caption
    s/\|left//ig;
    s/\|right//ig;
    s/\|\d+px//ig;
    s/\[\[image:[^\[\]]*\|//ig;
    s/\[\[category:([^|\]]*)[^]]*\]\]/[[$1]]/ig;  # show categories without markup
    s/\[\[[a-z\-]*:[^\]]*\]\]//g;  # remove links to other languages
    s/\[\[[^\|\]]*\|/[[/g;  # remove wiki url, preserve visible text
    s/\{\{[^}]*}}//g;         # remove {{icons}} and {tables}
    s/\{[^}]*}//g;
    s/\[//g;                # remove [ and ]
    s/\]//g;
    s/&[^;]*;/ /g;          # remove URL encoded chars

    # convert to lowercase letters and spaces, spell digits
    $_=" $_ ";
    #tr/A-Z/a-z/;
    s/0/ zero /g;
    s/1/ one /g;
    s/2/ two /g;
    s/3/ three /g;
    s/4/ four /g;
    s/5/ five /g;
    s/6/ six /g;
    s/7/ seven /g;
    s/8/ eight /g;
    s/9/ nine /g;
    s/  / /g;
    s/ \./\./g;
    s/ ,/,/g;
    s/'+/"/g;
    s/"s /'s/g;
    s/"t /'t/g;
    s/ '/'/g;
    s/=+/"/g;
    s/\//,/g;
    s/\"[ ]*\"/ /g;
    s/  / /g;
    s/"+/"/g;
    #s/" /"/g;
    #s/ "/"/g;
    s/ \)/\)/g;
    s/\( /\(/g;
    s/\n{4,}/\n\n\n/g;
    s/\*.*\n//g;
    s/:.*\n//g;
    s/#.*\n//g;
    s/"[ ]*books[ ]*"//gi;
    s/"[ ]*see also[ ]*"//gi; 
    s/"[ ]*references[ ]*"//gi; 
    s/"[ ]*trivia[ ]*"//gi;
    s/"[ ]*external links[ ]*"//gi;
    s/"[ ]*footnotes[ ]*"//gi;
    s/"[ ]*bibliography[ ]*"//gi;
    s/"[ ]*biographies[ ]*"//gi;
    s/"[ ]*external links[ ]*"//gi;
    s/[\{\}]//g;
    s/\n[ ]*/\n/g;
    s/[ ]*\n/\n/g;
    s/\n{3,}/\n\n/g;
    #tr/a-z/ /cs;
    s/[^$lowercase$uppercase$other]//g;
    chop;
    print output $_;
  }
}
