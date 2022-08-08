import pickle
import linecache
import sys
import faiss
import os
import pandas as pd
from rank_bm25 import *
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
from flask import request

from whoosh.index import create_in
from whoosh.fields import *
import shutil
from whoosh import qparser
from whoosh.qparser import QueryParser
from whoosh import scoring
from whoosh.index import open_dir


from tokenize import Number
from pyecharts.globals import CurrentConfig, NotebookType 
from pyecharts.charts import Bar
from pyecharts import options as opts
octis_path = "/home/mchenyu/2022sp/OCTIS"
sys.path.append(octis_path)
import pandas
import numpy as np
from octis.dataset.dataset import Dataset
from flask import Flask,render_template, request, redirect, url_for
from flask_table import Table, Col
bert_model = "bert-base-nli-mean-tokens"
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(bert_model)

from py4j.java_gateway import JavaGateway

app = Flask(__name__)
dataset = Dataset()
# path of all required data files, preprocessed by training model
dataset.load_custom_dataset_from_folder('/home/aman/topic_models_embeddings/ETDS_None')
path_model = "/home/aman/topic_models_embeddings/models/"
clean_data_path ="/home/aman/topic_models_embeddings/preprocessed_None.txt"
path_title = "/home/aman/topic_models_embeddings/titles_None.txt"
path_uri =  "/home/aman/topic_models_embeddings/uris_None.txt"
path_year = "/home/aman/topic_models_embeddings/years_None.txt"
path_university = "/home/aman/topic_models_embeddings/univs_None.txt"
path_university_list = "/home/aman/topic_models_embeddings/university_list_None.txt"
path_lucene = "/home/aman/topic_models_embeddings/lucene_data_None.txt"

pageNum = 0
bert_model = "bert-base-nli-mean-tokens"
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(bert_model)



# generate the table header
def top():
    html = "<div class=\"container\">\n"
    html += "  <ul class=\"responsive-table\">\n"
    html += "    <li class=\"table-header\">\n"
    html +=  "      <div class=\"col col-2\">Title</div>\n"
    html +=  "      <div class=\"col col-3\" style = \"text-align: left;\" >Abstract</div>\n"
    html +=  "    </li>"
    return html
# generate the table content
def create_table(title,abstract):
    html = ""
    for i in range(len(title)):
        html += "    <li class=\"table-row\">\n"
        html += "      <div class=\"col col-2\" data-label=\"Title\">"+title[i]+"</div>\n"
        html += "      <div class=\"col col-3 content hideContent\" data-label=\"Abstract\">"+abstract[i]+"</div>\n"
        html += "      <div class=\"show-more\"><button class = \"button-3\" type=\"button\">Show more</button></div>"
        html += "    </li>"
    return html
# generate the table footer
def end():
    return "  </ul>\n</div>"

#turning test to vector
def get_query_vector(query, embedding_types):
    vectors = {}
    query_list = [query]
    for emb in embedding_types:
        if emb == 'bert':
            vec = model.encode(query_list)
        if emb.startswith('random'):
            dim = int(emb.split('_')[1])
            vec = np.random.uniform(-1, 1, [1, dim])
        vectors[emb] = vec[0]
    return vectors


# Declare your table
class SingleItemTable(Table):
    description = Col('Topic')

# Get some objects
class SingleItem(object):
    def __init__(self, description):
        self.description = description


# Declare your table
class ItemTable(Table):
    name = Col('ID')
    description = Col('Topic')

# Get some objects
class Item(object):
    def __init__(self, name, description):
        self.name = name
        self.description = description

# Declare your table
class ItemNewTable(Table):
    name = Col('Title')
    description = Col('Abstract')

# Get some objects
class ItemNew(object):
    def __init__(self, name, description):
        self.name = name
        self.description = description

class FaissNN:
    def __init__(self, dim, db):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)   # build the index
        self.index.add(db)
        
    def get_nn(self, query, k):
        if len(query.shape) == 1:
            query = np.expand_dims(query, axis=0)
        D, I = self.index.search(query, k)
        return I, D


# index page
@app.route('/')
def index():
    return render_template('index.html')


# topic page which includes bar chart, containing a list of topics which are ranked
# by the number of documents in each topic.
@app.route('/<name>',methods = ['POST', 'GET'])
def success(name):
    if(name != 'favicon.ico'):
        web = 'http://localhost:5000/'+ name
        #getting the threshold from the user input to update the bar chart. 

        topform = '<form action = "' + web + '" method = "post" label for="threshold" style="font-size: larger; text-align: left;">&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Enter threshold: </label><input type="text" name="ts" id="threshold">&emsp;<button class = "button-3" type="submit">Update Threshold</button></form> </br>'
        if request.method == 'POST':
            threshold = float(request.form['ts'])
        else:
            threshold = 0.3

        # open the matched topic and numebr files and ranked by number of documents. 
        path = 'custom_None_' + name + '/output.p'
        with open(path_model + path, 'rb') as f:
            output = pickle.load(f)
        outputlist = []
        outputTopics = output['topics']
        topic_doc_matrix = output['topic-document-matrix']
        docs_per_topic = np.count_nonzero(topic_doc_matrix > threshold, axis=-1)
        new_topic_ids = np.argsort(-docs_per_topic)
        nameList = name.split('_')
        num_topics = int(nameList[1])
        # make the topic list hyperlink and redirect to the documents list page.
        for i in new_topic_ids:
            index = str(i+1)
            index = "<a href= \""+url_for('thirdPage',selected = name, number = index)+" \" >"+" ".join(outputTopics[i])+"</a>"
            outputlist += [dict(description=index)]

        
        # build the tables. 
        items = outputlist
        table = SingleItemTable(items)
        tableHTML = table.__html__()
        tableHTML = tableHTML.replace("&lt;","<")
        tableHTML = tableHTML.replace("&gt;",">")
        tableHTML = tableHTML.replace("&#34;","")
        fancyTable = '<div class="container"> <ul class="responsive-table"> <li class="table-header"> <div class="col col-extra">Topic</div></li>'
        index0 = tableHTML.find("<tr><td><a")
        tableHTML1 = (tableHTML[index0:])
        index1 = tableHTML1.find("</tbody>")
        tableHTML2 = (tableHTML1[:index1])
        tableHTML2 = tableHTML2.replace('<tr><td>', '<li class="table-row"><div class="col col-1" data-label="Topic">')
        tableHTML2 = tableHTML2.replace('</td></tr>', '</div></li>')
        fancyTable += tableHTML2
        fancyTable += '</ul> </div>'
        nameList = name.split("_")
        # add titles and fill the table.
        title = '<p style="font-family:\'Open Sans\'; font-weight: bold; text-align:left; color:Black; font-size: 30px;">'
        title += nameList[0]
        title += " Topic model with "
        title += nameList[1]
        title += ' topics </p>'
        # build the bar chart(removing the html header and footer)
        bar = (
            Bar()
            .add_xaxis([i for i in range(num_topics)])
            .add_yaxis("ETDs", docs_per_topic[new_topic_ids].tolist())
            .set_global_opts(title_opts=opts.TitleOpts(title="ETDs by Topic"))
        )
        temp = bar.render_embed()
        index = temp.find('<div id')
        tempfront = temp[index:]
        tempfront = tempfront.replace("</body>","")
        tempfront = tempfront.replace("</html>","")
        tempfront = tempfront.replace("width:900px; height:500px;","width:1200px; align-items:center; height:500px;")

        return render_template('index.html', t = title, desk = fancyTable, barchart = tempfront, newForm = topform)
    return render_template('index.html')

# Document page which includes titile, abstract and full Text URI
@app.route('/Doc/<num>/<document>',methods = ['POST', 'GET'])
def documentPage(num, document):
    number = num
    document = int(document)
    titles = linecache.getline(path_title, int(dataset.get_indexes()[document])+1)
    URIs = linecache.getline(path_uri, int(dataset.get_indexes()[document])+1)
    years = linecache.getline(path_year, int(dataset.get_indexes()[document])+1)
    university = linecache.getline(path_university, int(dataset.get_indexes()[document])+1)
    title_len = len(titles)
    abstracts = linecache.getline(clean_data_path, int(dataset.get_indexes()[document]) + 1)[title_len:]
    # replace the black titile with No title
    if titles == '':
        titles = 'No Title'
    # append the abstract
    subtitle = '<p style="font-family:Arial; font-weight: bold; text-align:center; color:Black; font-size: 30px;">'
    subtitle += str(titles)
    subtitle += ' </p>'
    p = '<p style="font-family:\'Open Sans\'; color:Black; font-size: 18px;"> '
    p += str(abstracts)
    p += '  </p>'
    back = int(num.split('_')[1])


    with open(path_model +'/custom_None_' + num +'/output.p', 'rb') as f:
            output = pickle.load(f)
    doc_topic_matrix = np.float32(output['topic-document-matrix'].transpose())
    if not doc_topic_matrix.data.c_contiguous:
        doc_topic_matrix = np.ascontiguousarray(doc_topic_matrix)
    NN = FaissNN(int(back), doc_topic_matrix)
    doc_id = int(document)
    I, D = NN.get_nn(doc_topic_matrix[doc_id], 6)
    ao = '<p style="font-family:\'Open Sans\'; font-weight: bold; text-align:left; color:Black; font-size: 20px;"> Similar documents:'
    r = '<p style="font-family:\'Open Sans\'; font-weight: bold; text-align:left; color:Black; font-size: 20px;"> Related Topics:'
    relatedTopic = '<p style="font-family:\'Open Sans\'; text-align:left; color:Black; font-size: 20px;">'
    o = '<p style="font-family:\'Open Sans\'; text-align:left; color:Black; font-size: 20px;"> '
    chao = '<a href="' + URIs + '">' + '<button class="button-3">Full Text URI</button></a>'


    path = 'custom_None_' + num + '/output.p'
    with open(path_model + path, 'rb') as f:
        outputf = pickle.load(f)
    outputlist = []
    outputTopics = outputf['topics']
    
    # find the top 3 most related topic list.
    doc_topics = (-doc_topic_matrix[document]).argsort()[:3]
    ind = 1
    for t in doc_topics:
        indexOfSecondPage = outputTopics.index(output['topics'][t])
        relatedTopic += "<a href= \""+url_for('thirdPage',selected = num, number = indexOfSecondPage + 1)+" \" >"+(','.join(output['topics'][t]))+"</a>"
        relatedTopic += ' <br />'
    
    # find the top most related document
    for i in I[0][1:]:
        title = linecache.getline(path_title, int(dataset.get_indexes()[i])+1)
        uri = linecache.getline(path_uri, int(dataset.get_indexes()[i])+1)
        year = linecache.getline(path_year, int(dataset.get_indexes()[i])+1)
        university = linecache.getline(path_university, int(dataset.get_indexes()[i])+1)
        if title == '':
            title = 'No Title'
        o +=  "<a href= \""+url_for('documentPage',num = number, document = str(i))+" \" >"+ title +"</a>"
        o += '<br />'
    return render_template('document.html', t = subtitle, ab = p, link = chao, aox = ao, rtitle = r, related = relatedTopic, other = o)


# documents list page by specific topic
@app.route('/<selected>/<number>',methods = ['POST', 'GET'])
def thirdPage(selected,number):
    # flag = int(flag)
    web = 'http://localhost:5000/'+ selected + '/' + number
    # create the form of searching start year, end year, threshold and university. 
    temporary = ''
    g = open(path_university_list,'r') 
    for line in g: 
        temporary += '<option value="' + line + '">'+ line + '</option>'
    topform = '<form action = "' + web + '" method = "post" label for="start year" style="font-family:\'Open Sans\';font-size:25px; text-align: left;">Start Year:</label><input type="text" name="sy" id="start year">&emsp;'
    topform += '<form action = "' + web + '" method = "post" label for="end year" style="font-family:\'Open Sans\';font-size:25px; text-align: left;">End Year:</label>'
    topform += '<input type="text" name="ey" id="end year">&emsp;<form action = "' + web + '" method = "post" label for="threshold" style="font-size: larger; text-align: left;">Threshold:</label><input type="text" name="ts" id="threshold">&emsp;'
    topform += '<form action = "' + web + '"method = "post" label for="TMD" style="font-family:\'Open Sans\';font-size:25px;font-size: larger; text-align: left;">University:</label><select name="university" id="TMD"  style="font-size: large; text-align: middle;"><option value="None">Please select a University</option>'
    topform += temporary
    topform += '</select>'

    topform += '&emsp; <form action = "' + web + '" method="post" style="font-family:\'Open Sans\';font-size:25px; position: right; display: inline-block;text-align: right;"><!-- <input type="text" placeholder="Search Key Words.." name="search" > --><button class = "button-3" type="submit">Filter Documents</button></form>'

    # Search page(getting the parameter of start year, end year, threshold and university.)
    if request.method == 'POST':
        topic_id = int(number) - 1 
        start_year = int(request.form['sy'])
        end_year = int(request.form['ey'])
        threshold = float(request.form['ts'])
        path = 'custom_None_' + selected + '/output.p'
        with open(path_model + path, 'rb') as f:
            output = pickle.load(f)
        topic_doc_matrix = output['topic-document-matrix']
        doc_indexes = np.where(topic_doc_matrix[topic_id] > threshold)[0]
        year_counts = {y:0 for y in range(start_year, end_year+1)}
        dclist = []
        count = 0
        selected_university = request.form['university'].strip()
        titlelist = []
        abstractlist = []
        # filter the documents by the given condition(start year, end year, threshold and university). 
        for doc_idx in doc_indexes:
            titles = linecache.getline(path_title, int(dataset.get_indexes()[doc_idx])+1)
            URIs = linecache.getline(path_uri, int(dataset.get_indexes()[doc_idx])+1)
            years = linecache.getline(path_year, int(dataset.get_indexes()[doc_idx])+1)
            university = linecache.getline(path_university, int(dataset.get_indexes()[doc_idx])+1)
            title_len = len(titles) 
            titles = "<a href= \""+url_for('documentPage', num = selected, document = doc_idx)+" \" >"+ titles+"</a>"
           
            abstracts = linecache.getline(clean_data_path, int(dataset.get_indexes()[doc_idx])+1)[title_len:]
            if(years == 'None\n' or years == ''):
                continue
            year = int(years.split('.')[0])
            if year >= start_year and year <= end_year and university.strip() == selected_university:
                count += 1
                titlelist += [titles]
                abstractlist += [abstracts]
        
        # create the tables for the document list. 
        finalTable = top()
        finalTable += create_table(titlelist,abstractlist)
        finalTable += end()
        finalTable = finalTable.replace("<p>", "").replace("</p>", "")
        subtitle = '<p style="font-family:\'Open Sans\'; font-weight: bold; text-align:center; color:Black; font-size: 30px;"> Found '
        subtitle += str(count)
        subtitle += ' related Documents </p>'

        year_counts = {y:0 for y in range(start_year, end_year+1)}
        for doc_idx in doc_indexes:
            titles = linecache.getline(path_title, int(dataset.get_indexes()[doc_idx])+1)
            URIs = linecache.getline(path_uri, int(dataset.get_indexes()[doc_idx])+1)
            years = linecache.getline(path_year, int(dataset.get_indexes()[doc_idx])+1)
            university = linecache.getline(path_university, int(dataset.get_indexes()[doc_idx])+1)


            if(years == 'None\n' or years == ''):
                continue
            year = int(years.split('.')[0])
            if year >= start_year and year <= end_year and university.strip() == selected_university:
                year_counts[year] = year_counts[year]+1
        # creating the bar for the filtered document list
        bar = (
            Bar()
            .add_xaxis(list(year_counts.keys()))
            .add_yaxis("ETDs", list(year_counts.values()))
            .set_global_opts(title_opts=opts.TitleOpts(title="ETDs by Year"))
        )
        temp = bar.render_embed()
        index = temp.find('<div id')
        tempfront = temp[index:]
        tempfront = tempfront.replace("</body>","")
        tempfront = tempfront.replace("</html>","")
        tempfront = tempfront.replace("width:900px;","width:900px; align-items:center;")
        

        return render_template('advanced_searched.html', top = topform, sub = subtitle, barchart = tempfront, desk = finalTable)

    # get the page of regular document list
    if request.method == 'GET':
        path = 'custom_None_' + selected + '/output.p'
        with open(path_model + path, 'rb') as f:
            output = pickle.load(f)
        topic_id = int(number) - 1 #topics index
        top_k = 10
        topic_doc_matrix = output['topic-document-matrix']
        top_docs = (-topic_doc_matrix[topic_id]).argsort()[:top_k]

        dclist = []
        titlelist = []
        abstractlist = []
        for doc_idx in top_docs:
            x = linecache.getline(path_title, int(dataset.get_indexes()[doc_idx])+1)
            university = linecache.getline(path_university, int(dataset.get_indexes()[doc_idx])+1)
            if x == '':
                x = 'No title'

            title_len = len(x)
            x = "<a href= \""+url_for('documentPage',num = selected, document = doc_idx)+" \" >"+ x+"</a>"
            a = linecache.getline(clean_data_path, int(dataset.get_indexes()[doc_idx])+1)[title_len:]
            titlelist += [x]
            abstractlist += [a]
            
        finalTable = top()
        finalTable += create_table(titlelist,abstractlist)
        finalTable += end()
        finalTable = finalTable.replace("<p>", "").replace("</p>", "")

        content = '<p style="font-family:\'Open Sans\'; font-weight: bold; text-align:center; color:Black; font-size: 30px;">'
        content += 'Topic Words: '
        content += str(output['topics'][topic_id]).replace('[', '').replace(']', '')
        content += '</p>'

        outputTopics = output['topics']
        relatedTopic = '<p style="font-family:\'Open Sans\'; text-align:left; font-weight: bold; color:Black; font-size: 20px;"> Similar Topics: </br>'
        
        # get the top 3 most revelant topics. 
        vocab_size = len(dataset.get_vocabulary())
        topicNN = FaissNN(vocab_size, output['topic-word-matrix'])
        top_k = 3
        K, _ = topicNN.get_nn(output['topic-word-matrix'][topic_id], top_k+1)
        for t in K[0][1:]:
            indexOfSecondPage = t
            relatedTopic += "<a href= \""+url_for('thirdPage',selected = selected, number = indexOfSecondPage + 1)+" \" >"+(','.join(output['topics'][t]))+"</a>"
            relatedTopic += ' <br />'
        return render_template('advanced_searched.html', top = topform, title = content, desk = finalTable, similar = relatedTopic)


# search result page given keywords and search method
@app.route('/Search/<model_type>/<keyWord>/<method>',methods = ['POST', 'GET'])
def searchPage(keyWord,method,model_type):
    ix = open_dir("indexdir")
    global pageNum
    print(pageNum)
    if request.method == 'POST': 
        if(request.form["nextPage"] == 'update'):
            pageNum += 1
        print(request.form["nextPage"])
    else:
        pageNum = 1
        print("Enter the else")
    
    titlelist = []
    abstractlist = []
    # connect the JVM
    gateway = JavaGateway()

    if(method == 'BM25'):
        # envoke the java lucene search method to get the document id list
        jList = gateway.entry_point.search("BM-25", 10, keyWord)
        # tuen the java arraylist to python list. 
        pyList = list(jList)
        # gateway.shutdown()
        for hit in pyList:
                doc_idx = int(hit)
                print(int(dataset.get_indexes()[doc_idx]))
                
                titles = linecache.getline(path_lucene, int(doc_idx)+1).split('\t')[1]
                title_len = len(titles)
                titles = "<a href= \""+url_for('documentPage', num = model_type, document = doc_idx - 1)+" \" >"+ titles+"</a>"
                abstract = linecache.getline(path_lucene, int(doc_idx)+1).split('\t')[5]
                titlelist += [titles]
                abstractlist += [abstract]


        # with ix.searcher(weighting=scoring.BM25F()) as searcher:
        #     parser = QueryParser("content", ix.schema, group=qparser.OrGroup)
        #     query_terms = keyWord
        #     query = parser.parse(query_terms)
        #     print(query)
        #     results = searcher.search(query, limit=None)
        #     print("Found {} matching documents".format(str(len(results))))
        #     batch_no = pageNum
        #     page_results = searcher.search_page(query, batch_no)
        #     for hit in page_results:
        #         doc_idx = int(hit['doc_id'])
        #         titles = linecache.getline(path_title, int(dataset.get_indexes()[doc_idx])+1)
        #         title_len = len(titles)
        #         titles = "<a href= \""+url_for('documentPage', num = model_type, document = doc_idx)+" \" >"+ titles+"</a>"
        #         abstract = linecache.getline(clean_data_path, int(dataset.get_indexes()[doc_idx])+1)[title_len:]
        #         titlelist += [titles]
        #         abstractlist += [abstract]
                

    if(method == 'TF-IDF'):
        jList = gateway.entry_point.search("TF-IDF", 10, keyWord)
        pyList = list(jList)
        # gateway.shutdown()
        for hit in pyList:
                doc_idx = int(hit)
                print(doc_idx)
                # int(dataset.get_indexes()[doc_idx])+1
                titles = linecache.getline(path_lucene, int(doc_idx)+1).split('\t')[1]
                title_len = len(titles)
                titles = "<a href= \""+url_for('documentPage', num = model_type, document = doc_idx - 1)+" \" >"+ titles+"</a>"
                abstract = linecache.getline(path_lucene, int(doc_idx)+1).split('\t')[5]
                titlelist += [titles]
                abstractlist += [abstract]

    # vector search method
    if(method == 'Vector Search'):
        query = keyWord
        print(keyWord)
        get_query_vector(query, ['bert'])
        array = get_query_vector(query, ['bert'])['bert']
        result = array.tolist()
        res2 = " ".join(map(str, result))
        jList = gateway.entry_point.testWriteAndQueryIndex(res2)
        pyList = list(jList)
        print("This is the length of the jList")
        print(len(pyList))
        for hit in pyList:
                doc_idx = int(hit) + 1
                print(doc_idx)
                # int(dataset.get_indexes()[doc_idx])+1
                titles = linecache.getline(path_lucene, int(doc_idx)+1).split('\t')[1]
                title_len = len(titles)
                titles = "<a href= \""+url_for('documentPage', num = model_type, document = doc_idx - 1)+" \" >"+ titles+"</a>"
                abstract = linecache.getline(path_lucene, int(doc_idx)+1).split('\t')[5]
                titlelist += [titles]
                abstractlist += [abstract]    
    

    

    




        # with ix.searcher(weighting=scoring.TF_IDF()) as searcher:
        #     parser = QueryParser("content", ix.schema, group=qparser.OrGroup)
        #     query_terms = keyWord
        #     query = parser.parse(query_terms)
        #     print(query)
        #     results = searcher.search(query, limit=None)
        #     print("Found {} matching documents".format(str(len(results))))
        #     batch_no = pageNum
        #     page_results = searcher.search_page(query, batch_no)
        #     print(page_results)
        #     for hit in page_results:
        #         doc_idx = int(hit['doc_id'])
        #         print(doc_idx)
        #         titles = linecache.getline(path_title, int(dataset.get_indexes()[doc_idx])+1)
        #         title_len = len(titles)
        #         titles = "<a href= \""+url_for('documentPage', num = model_type, document = doc_idx)+" \" >"+ titles+"</a>"
        #         abstract = linecache.getline(clean_data_path, int(dataset.get_indexes()[doc_idx])+1)[title_len:]
        #         titlelist += [titles]
        #         abstractlist += [abstract]
    
    finalTable = top()
    finalTable += create_table(titlelist,abstractlist)
    finalTable += end()
    finalTable = finalTable.replace("<p>", "").replace("</p>", "")
    return render_template('next.xhtml', desk = finalTable)


def get_query_vector(query, embedding_types):
    vectors = {}
    query_list = [query]
    for emb in embedding_types:
        if emb == 'bert':
            vec = model.encode(query_list)
        if emb.startswith('random'):
            dim = int(emb.split('_')[1])
            vec = np.random.uniform(-1, 1, [1, dim])
        vectors[emb] = vec[0]
    return vectors

# method of getting parameter from the front end
# includes the topic model and number, keywords and method for searching. 
@app.route('/',methods = ['POST', 'GET'])
def login():
    if request.method == 'POST':
        if request.form['action'] == 'first':
            to = request.form['ab']
            user = request.form['nm']
            return redirect(url_for('success',name = to + '_' + user))
        if request.form['action'] == 'second':
            print('enter the second')
            to = request.form['ab']
            user = request.form['nm']
            model_type = to + '_' + user
            key = request.form['Keyword']
            meth = request.form['Method']
            return redirect(url_for('searchPage',keyWord = key, method = meth, model_type=model_type))
    else:
        if request.form['action'] == 'first':
            to = request.form['ab']
            user = request.form['nm']
            return redirect(url_for('success',name = to + '_' + user))
        if request.form['action'] == 'second':
            to = request.form['ab']
            user = request.form['nm']
            model_type = to + '_' + user

            key = request.form['Keyword']
            meth = request.form['Method']
            return redirect(url_for('searchPage',keyWord = key, method = meth, model_type=model_type))

