import os
import spacy
import pickle
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from dataset_parse_util import parse_sp
from dataset_parse_util import traverse_all_children
from dataset_parse_util import traverse_all_parent
from spacy.matcher import Matcher
from fuzzywuzzy import fuzz
import ast
from spacy import displacy
from pathlib import Path
import matplotlib.pyplot as plt

np.random.seed(42)
nlp = spacy.load("en_core_web_sm")
nlp.disable_pipes("ner")
nlp.add_pipe(nlp.create_pipe("merge_noun_chunks"))


matcher = Matcher(nlp.vocab)
pattern = [{'IS_ALPHA': True, 'IS_SPACE': False},
           {'ORTH': '-'},
           {'IS_ALPHA': True, 'IS_SPACE': False}]

matcher.add('HYPHENATED', None, pattern)

def parse_with_displacy(sen):

    api = "http://localhost:8000"
    
    svg = displacy.render(nlp(sen), style="dep", options= {  'distance':140,'font':10})
    output_path = Path("C:/Users/rana/source/repos/CauseEffectExtraction/CauseEffectExtraction/images/img.svg")
    output_path.open("w", encoding="utf-8").write(svg)
    i=0

def quote_merger(doc):
    # this will be called on the Doc object in the pipeline
    matched_spans = []
    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        matched_spans.append(span)
    for span in matched_spans:  # merge into one token after collecting all matches
        span.merge()
    #print(doc)
    return doc

def find_next(noun_chunks, current_pair, other_pair, sentence):
    noun_chunks2 = [str(a).lower() for a in noun_chunks] 
    k = noun_chunks2.index(current_pair)
    temp = sentence
    prev_index = temp.index(str(current_pair).lower())
    try:
        if k < len(noun_chunks2) - 1:
            k = k + 1
            next_np = str(noun_chunks[k])
            if(next_np == str(other_pair).lower()):
                return None
            next_index = temp.index(next_np.lower())
            span_index = prev_index + len(current_pair)
            if span_index < next_index:
                check_conj = temp[span_index:next_index-1]
                check_conj = check_conj.strip()
                if(check_conj == 'and' or check_conj == 'or' or check_conj == 'such as' or check_conj == 'including' or check_conj == ',' or check_conj == ', and' or check_conj == ', or'):
                    if (noun_chunks[k].dep_ == noun_chunks[k-1].dep_) or noun_chunks[k].dep_ == 'conj':
                        return next_np
                    else:
                        return None
                else:
                    return None
        else:
            None
    except:
        None

def get_patterns():
    seed_patterns = pd.read_csv(os.path.join(os.path.abspath(""), "data", "seed_patterns_ce.tsv"), delimiter="\t", header=None).values

    patterns = []

    for seed_pattern in seed_patterns:
        _, _, zs, _ = seed_pattern
        for i,z in enumerate(zs.split("|")):
            patterns.append(z)
    patterns = list(set(np.array(patterns)))
    t = np.array([len(p.split()) for p in patterns])
    idx = t.argsort()[::-1]
    patterns = np.array(patterns)[idx]
    return patterns

def clean_phrase(x,sen,patterns):
    sen = sen.lower()
    x = x.lower()
    for p in patterns:
        x = x.lower()
        p = p.lower()
        k = p + ' '
        if x.lower().find(k.lower()) > -1:
            x = str(x[x.find(k)+len(k):]).strip()
    for p in patterns[10:]:
        x = x.lower()
        k = p.lower().split()[0] + ' '
        if x.lower().find(k.lower()) > -1:
            x = str(x[x.find(k)+len(k):]).strip()

    if x.find(" and") > len(x) - 6:
        x = x.replace("and","")
        x = x.strip()

    if x.find(" or") > len(x) - 5:
        x = x.replace("or","")
        x = x.strip()

    stopwords = ["by ", "with "]

    for st in stopwords:

        if x.find(st) == 0:
            x = x.replace(st,"")
            x = x.strip()

    if x.find("are ") == 0:
        x = x[x.find('are ') + len('are '):]

    if x.find("were ") == 0:
        x = x[x.find('are ') + len('are '):]

    if x.find("the products of ") == 0:
        x = x[x.find('the products of ') + len('the products of '):]

    if x.find("the results of ") == 0:
        x = x[x.find('the results of ') + len('the results of '):]

    if sen.find(x) == -1:
        ind_start = sen.find(" ".join(x.split()[:2]))
        if ind_start == -1:
            ind_start = sen.find(" ".join(x.split()[:1]))
        ind_end = sen.find(" ".join(x.split()[-2:]))
        if ind_end == -1:
            if sen.find(x.split()[-1]) > sen.find(x.split()[-2]):
                ind_end = sen.find(" ".join(x.split()[-1:])) + len(" ".join(x.split()[-1:]))
            else:
                ind_end = sen.find(x.split()[-2]) + len(x.split()[-2])
        else:
            ind_end = ind_end + len(" ".join(x.split()[-2:]))
        #if ind_start
        if ind_start < ind_end:
            check = sen[ind_start:ind_end]
            x = check
    stopwords = [" such", " and"]

    for st in stopwords:
        if x.find(st) > -1 and x.find(st) + len(st) > len(x) - 1:
            x = x.replace(st,"")
            x = x.strip()

    if x == 'that' or x == 'which' or x == 'they' or x =='who':
        return None

    if x.find("that") > 0:
        x = x[:x.find('that')]

    if x.find("as ") == 0:
        x = x[x.find('as ') + len('as '):]

    if x.find("of ") == 0:
        x = x[x.find('of ') + len('of '):]

    if x.find("to ") == 0:
        x = x[x.find('to ') + len('to '):]

    if x.find("from ") == 0:
        x = x[x.find('from ') + len('from '):]

    x = x.strip()

    return x



def find_prev(noun_chunks, current_pair, other_pair,sentence):
    noun_chunks2 = [str(a).lower() for a in noun_chunks] 
    k = noun_chunks2.index(current_pair)
    temp = sentence
    next_index = temp.index(str(current_pair).lower())
    try:
        if k > 0:
            k = k - 1
            prev_np = str(noun_chunks[k])
            if(prev_np == str(other_pair).lower()):
                return None
            prev_index = temp.index(prev_np.lower())
            span_index = prev_index + len(prev_np)
            if span_index < next_index:
                check_conj = temp[span_index:next_index-1]
                check_conj = check_conj.strip()
                if(check_conj == 'and' or check_conj == 'or' or check_conj == ',' or check_conj == ', and' or check_conj == ', or'):
                    if (noun_chunks[k].dep_ == noun_chunks[k+1].dep_) or noun_chunks[k+1].dep_ == 'conj':
                        return prev_np
                    else:
                        return None
                else:
                    return None
        else:
            None
    except Exception as ex:
        print(ex)
        None


def generate_patterns():
    patterns = {}
    for fi in ["ce"]:
        patterns[fi] = []
        seed_patterns = pd.read_csv(
            os.path.join(os.path.abspath(""), "data", "seed_patterns_{}.tsv".format(fi)), delimiter="\t", header=None).values

        for i,seed_pattern in enumerate(seed_patterns):
            x, y, zs, sentence = seed_pattern

            for z in zs.split("|"):
                try:
                    tmp_sentence = sentence.replace(
                        "X", x).replace("Y", y).replace("Z", z)
                    tmp_sentence.replace(
                        tmp_sentence[0], tmp_sentence[0].upper())

                    doc = nlp(tmp_sentence)
                    edges = parse_sp(x, y, doc, nlp)
                    edges = [",".join(edge) for edge in edges]

                    if edges not in patterns[fi] and len(edges) > 1:
                        patterns[fi].append(edges)

                except Exception as e:
                    pass

    return patterns

def unique(list1): 
  
    # intilize a null list 
    unique_list = [] 
      
    # traverse for all elements 
    for x in list1: 
        # check if exists in unique_list or not 
        if x not in unique_list: 
            unique_list.append(x)
    return unique_list


def pattern_intersect(edges, patterns, threshold=1.0):
    for i, pattern in enumerate(patterns):
        t = len(list(set(edges).intersection(pattern))) / len(pattern) 
        #print(t)
        if t >= threshold:
            return pattern
    return None


def parse(patterns_all, raw_sentences):
    rel_pos, rel_neg = {}, {}
    #input = args["input_file"]
    filtered_sentences = []

    keywords = ["COVID-19","Coronavirus","Corona virus","2019-nCoV",
                "SARS-CoV","MERS-CoV","SARS","MERS","contagious", "contagion"]

    for raw in raw_sentences:

        
        try:
            doc = nlp(raw)
            sentences = [sent.string.strip() for sent in doc.sents]

            for sentence in sentences:

                if sentence.find('prompted')>0:
                    ll = 9

                doc = nlp(sentence)
                noun_chunks = []#list(doc.noun_chunks)

                for tok in doc:
                    if tok.pos_ == 'NOUN' or tok.pos_ =='PROPN' or tok.pos_ =='PRON':
                        if tok not in noun_chunks:
                            noun_chunks.append(tok)

                temp = str(sentence).lower()

                for i, x in enumerate(noun_chunks):
                    for j, y in enumerate(noun_chunks):

                        if i == j:
                            continue
                        try:
                            if i < j:
                                if temp.index(str(x).lower()) + len(str(x)) > temp.index(str(y).lower()) - 5:
                                    continue
                            else:
                                if temp.index(str(y).lower()) + len(str(y)) > temp.index(str(x).lower()) - 5:
                                    continue
                        except:
                            continue

                        edges = parse_sp(
                            x.lower_, y.lower_, doc, nlp)
                        edges = [",".join(edge) for edge in edges]

                        patt_ce = pattern_intersect(edges, patterns["ce"])

                        if patt_ce is not None:
                            if "not" not in str(sentence):
                                xs = []
                                xs.append(str(x).lower())
                                t = find_next(noun_chunks,str(x).lower(),str(y).lower(),str(sentence).lower())
                                while t != None:
                                    xs.append(t)
                                    t = find_next(noun_chunks, t ,str(y).lower(),str(sentence).lower())

                                t = find_prev(noun_chunks,str(x).lower(),str(y).lower(),str(sentence).lower())
                                while t != None:
                                    xs.append(t)
                                    t = find_prev(noun_chunks, t ,str(y).lower(),str(sentence).lower())


                                ys = []
                                ys.append(str(y).lower())
                                t = find_prev(noun_chunks,str(y).lower(),str(x).lower(),str(sentence).lower())
                                while t != None:
                                    ys.append(t)
                                    t = find_prev(noun_chunks, t ,str(x).lower(),str(sentence).lower())

                                t = find_next(noun_chunks,str(y).lower(),str(x).lower(),str(sentence).lower())
                                while t != None:
                                    ys.append(t)
                                    t = find_next(noun_chunks, t ,str(x).lower(),str(sentence).lower())

                                for t in xs :
                                    filtered_sentences.append([t, y.lower_, sentence, str(patt_ce), "1"])
                                for t in ys :
                                    filtered_sentences.append([x.lower_, t, sentence, str(patt_ce), "1"])

        except Exception as e:
            print(e)
            pass

    pd.DataFrame(data = unique(filtered_sentences)).to_csv(os.path.join(os.path.abspath(""),"extracted_pairs", "predictions.csv"),index = None,header=None,columns =None)

def merge_phrases():
    df = pd.read_csv(os.path.join(os.path.abspath(""),"extracted_pairs", "predictions.csv"),header=None)
    values = df.values
    patterns = get_patterns()
    #nlp.add_pipe(quote_merger, first=True) 
    candidates = []

    for i,val in enumerate(values):
        x = traverse_all_children(str(val[0]).lower(),str(val[1]).lower(),nlp(str(val[2])),nlp)
        y = traverse_all_children(str(val[1]).lower(),str(val[0]).lower(),nlp(str(val[2])),nlp)
        if x == str(val[0]).lower():
            x = traverse_all_parent(str(val[0]).lower(),str(val[1]).lower(),nlp(str(val[2])),nlp)
        if y == str(val[1]).lower():
            y = traverse_all_parent(str(val[1]).lower(),str(val[0]).lower(),nlp(str(val[2])),nlp)

        try:
            x = clean_phrase(x,val[2],patterns)
            y = clean_phrase(y,val[2],patterns)
        except:
            i = 0
        
        if(x == '' or y == '' or val[2] == '' or val[3] == ''):
            continue

        if [x,y,val[2],val[3]] not in candidates and x != None and y != None:
            candidates.append([x,y,val[2],val[3]])

    pd.DataFrame(data=candidates).to_csv("spacy_extended.csv",header=None,index=None,columns=None)

def check_accuracy():
    
    micro_precision = 0
    macro_precision = 0
    micro_recall = 0
    macro_recall = 0

    #predictions = [(str(v[0]).lower().strip(),str(v[1]).lower().strip()) for v in pd.read_csv(os.path.join(os.path.abspath(""),"extracted_pairs", "predictions.csv"),header=None).values]

    predictions = [(str(v[0]).lower().strip(),str(v[1]).lower().strip()) for v in pd.read_csv("spacy_extended.csv",header=None).values]
    actuals = [(str(v[0]).lower().strip(),str(v[1]).lower().strip()) for v in pd.read_csv("concatatenated_train.csv",header=None).values if v[-1] == 1]

    common = 0

    for p in predictions:
        for a in actuals:
            if (p[0] in a[0] or a[0] in p[0]) and (p[1] in a[1] or a[1] in p[1]):
                common = common + 1
                #break

    precision = common / len(set(predictions))
    recall = common / len(set(actuals))
    print("Precision= "+str(precision))
    print("Recall= "+str(recall))

def check_accuracy_per_data(predictions,actuals):

    common = 0

    for p in predictions:
        for a in actuals:
            if (p[0] in a[0] or a[0] in p[0]) and (p[1] in a[1] or a[1] in p[1]):
                common = common + 1
                #break

    precision = common / len(set(predictions))
    recall = common / len(set(actuals))
    f1 = (2 * precision * recall) / (precision + recall)
    return [precision,recall,f1]


def draw_histogram():
    values = pd.read_csv("test_dataset.csv", header=None).values
    predicted= pd.read_csv("spacy_extended.csv",header=None).values
    predictions = [(str(v[0]).lower().strip(),str(v[1]).lower().strip()) for v in pd.read_csv("spacy_extended.csv",header=None).values]
    maxL = -1
    for i, v in enumerate(values):
        x = str(v[0])
        y = str(v[1])
        currentMax = max(len(x.split()),len(y.split()))
        if maxL < currentMax :
            maxL = currentMax


    pairs = []
    for i in range(maxL):
        pairs.append([])
    for i, v in enumerate(values):
        x = str(v[0])
        y = str(v[1])
        currentMax = max(len(x.split()),len(y.split()))
        temp = [x,y,str(v[2])]
        pairs[currentMax-1].append(temp)

    f1scores = []

    for chnk in pairs:
        actuals = [(str(v[0]).lower().strip(),str(v[1]).lower().strip()) for v in chnk]
        preds = []
        for p in predicted:
            for v in chnk:
                if len(v)<=2:
                    continue
                if str(p[2]).lower().strip() == str(v[2]).lower().strip():
                    preds.append((str(p[0]).lower().strip(),str(p[1]).lower().strip()))
        common = 0

        for p in preds:
            for a in actuals:
                if (p[0] in a[0] or a[0] in p[0]) and (p[1] in a[1] or a[1] in p[1]):
                    common = common + 1
                    #break
        try:
            precision = common / len(set(preds))
            recall = common / len(set(actuals))
            f1 = (2 * precision * recall) / (precision + recall)
            f1scores.append(f1)
        except:
            continue
    frequencies = [len(p) for p in pairs]

    temp_frequencies = frequencies[:14]
    temp_frequencies.append(np.sum(frequencies[14:]))

    frequencies = temp_frequencies


    alphab = list(range(len(frequencies)))

    alphab= [a+1 for a in alphab]

    alphab = [str(a) for a in alphab]

    alphab[-1] = '  >=15'

    pos = np.arange(len(alphab))
    width = 0.5     # gives histogram aspect to the bar diagram

    ax = plt.axes()
    ax.set_xticks(pos + (width / 2))
    ax.set_xticklabels(alphab)

    plt.xlabel("Length") 
    plt.ylabel("Frequency") 

    plt.bar(pos, frequencies, width,align='edge')
    plt.show()

    #plt.bar(x=range(len(lengths)), data=lengths)
    #plt.show()




def check_accuracy_per_rule():
    
    micro_precision = 0
    macro_precision = 0
    #micro_recall = 0
    #macro_recall = 0

    predictions = [[(str(v[0]).lower().strip(),str(v[1]).lower().strip()),ast.literal_eval(v[-1])] for v in pd.read_csv("spacy_extended.csv",header=None).values]
    actuals = [(str(v[0]).lower().strip(),str(v[1]).lower().strip()) for v in pd.read_csv("concatatenated_train.csv",header=None).values if v[-1] == 1]

    distinct_patterns = [p[1] for p in predictions]

    dummy = []

    [dummy.append(p) for p in distinct_patterns if p not in dummy]

    distinct_patterns = dummy

    result = []

    for p in distinct_patterns:
        current_pairs = [k[0] for k in predictions if k[1] == p]
        prec = len(set(current_pairs).intersection(set(actuals))) / len(set(current_pairs))
        intersections = 0

        for pk in current_pairs:
            for a in actuals:
                pk = list(pk)
                a = list(a)
                try:
                    if fuzz.ratio(a[0],pk[0])>60 and fuzz.ratio(a[1],pk[1])>60:
                        intersections = intersections + 1
                        break
                except:
                    continue
        result.append([prec,intersections/len(current_pairs),p,len(current_pairs)])
    pd.DataFrame(data = result).to_csv("precisions.csv",header=None,columns=None,index=None)

if __name__ == "__main__":

    #draw_histogram()
    patterns = generate_patterns()

    #raw_sentences = pd.read_csv("dummy.csv",header = None).values[:,2]
    raw_sentences = pd.read_csv("concatatenated_train.csv",header=None).values
    raw_sentences = [a[2] for a in raw_sentences if a[-1] == 1]

    #parse(patterns, raw_sentences)

    #merge_phrases()

    check_accuracy()

    check_accuracy_per_rule()
