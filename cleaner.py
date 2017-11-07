# -*- coding: utf-8 -*-
import re
import nltk

regex = re.compile('[a-zA-Z0-9\s\'/&]+\-[a-zA-Z0-9\s\'/&]+:')
# company_regex = re.compile('www\.[a-zA-Z]+\.com')
company_regex = re.compile('[a-zA-Z0-9\-]+\.[a-zA-Z0-9\-]+\.[a-zA-Z0-9\-]+')


def clean(data):
    company_matched = company_regex.search(data)
    matched = regex.match(data)


    data = re.sub('[^\x00-\x7F]+', ' ', data)
    #data = extract_entities(data)
    if matched and company_matched:
        data = data.replace(matched.group(), "").strip()
        data = data.replace(company_matched.group(), "").strip()

        data = extract_entities(data)
        return data
    elif matched:
        data = data.replace(matched.group(), "").strip()
        data = extract_entities(data)
        return data
    elif company_matched:
        data = data.replace(company_matched.group(), "").strip()
        # print company_matched.group()
        data = extract_entities(data)
        return data
    else:
        data = extract_entities(data)
        return data


def extract_entities(text):

    nnps = []
    for sent in nltk.sent_tokenize(text):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if isinstance(chunk,nltk.tree.Tree):

                 #if chunk._label == 'PERSON':
                 if chunk.node == 'PERSON':
                     #print 'Person'
                     #print ' '.join(c[0] for c in chunk.leaves())
                     #print "--------"
                     nnps.append(' '.join(c[0] for c in chunk.leaves()))
                 elif chunk.node == 'ORGANIZATION':
                     #print "Org"
                     #print ' '.join(c[0] for c in chunk.leaves())
                     #print "--------"
                     nnps.append(' '.join(c[0] for c in chunk.leaves()))
                 elif chunk.node == 'GPE':
                     #print "GPE"
                     #print ' '.join(c[0] for c in chunk.leaves())
                     #print "--------"
                     nnps.append(' '.join(c[0] for c in chunk.leaves()))
            elif isinstance(chunk,tuple):

                if chunk[1] == 'CD':
                    #print "CD"
                    #print ' '.join(chunk[0])
                    #print "--------"
                    #print chunk[0]
                    nnps.append(chunk[0])


    for each_nnp_word in nnps:

        text = text.replace(each_nnp_word, "").strip()

    #print text
    #print "************************************************************************"
    return text



if __name__ == '__main__':
    # clean("tructural Metal Fabrication - addtional description: please visit www.saks.com.")
    # print "-----------"
    # clean("www.biconet.com")
    # company_matched = company_regex.match("www.biconet.com")
    # print company_matched.group()
    print clean("Gajanand Granites is leading company engaged in processing and manufacturing of stone like marble ,granite ,sandstone ,slate ,cobbles ,pebbles ,step ,rizer ,stairs , monument etc from last decade . company is also exporting good quality material to many country with huge number of satisfied customer . the main aim of company is customer delight . Gajanand granite has capacity to supply any kind of material in any amout . we have proud that we completed many overseas project successfully . we are looking forward to establish long term forever relationship with esteemed customer like you")
