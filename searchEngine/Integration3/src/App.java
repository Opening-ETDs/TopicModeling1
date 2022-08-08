import java.io.*;
import java.lang.module.FindException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.ParseException;
import java.util.*;

import org.apache.commons.io.FileUtils;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene90.Lucene90Codec;
import org.apache.lucene.codecs.lucene90.Lucene90HnswVectorsFormat;
import org.apache.lucene.document.*;
import org.apache.lucene.document.Field.Store;
import org.apache.lucene.index.*;
import org.apache.lucene.queryparser.classic.MultiFieldQueryParser;
import org.apache.lucene.search.*;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.store.*;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.BytesRef;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;


//
import py4j.GatewayServer;


public class App {
    public static final Path indexPath = Paths.get("target/ind");
    public static final VectorSimilarityFunction similarityFunction = VectorSimilarityFunction.EUCLIDEAN;
    public static final int maxConn = 14;
    public static final int beamWidth = 5;
    private static newVectorProvider vectors;
    public static final float[] testData = new float[768];

    public static final String pathOfCSV = "/home/aman/topic_models_embeddings/lucene_data_bert_random_50_None.txt";
    public static final List<Vector<Float>> vector50 = new ArrayList<>();


    public static void init() throws IOException
        {
            File file = new File(pathOfCSV);
            int count = 0; 
            // CSVReader csvReader = new CSVReader(new FileReader(file));
            BufferedReader br = new BufferedReader(new FileReader(file));

            if (file.exists()) {
                StandardAnalyzer analyzer = new StandardAnalyzer();
                Directory index = FSDirectory.open(Paths.get("/home/mchenyu/22summer/Integration3/index1/documents.index"));

            IndexWriterConfig config = new IndexWriterConfig(analyzer);
            IndexWriter w = new IndexWriter(index, config);
            String nextLine;
            while ((nextLine = br.readLine()) != null) {
                
                String[] cols = nextLine.split("\t");
                String id = Integer.valueOf(count).toString(); 
                String title = cols[1];
                count++;
                // System.out.println(title);
                
                String abst = cols[5];
                // System.out.println(abst); 
                addDoc(w, id, title, abst);
            }
            br.close();
            w.close();
            // System.out.println(count); 
        }
    }

        public static List<String> search(String searchMode, int hitsPerPage, String query) throws IOException, ParseException, org.apache.lucene.queryparser.classic.ParseException
    {
        StandardAnalyzer analyzer = new StandardAnalyzer();
        Directory index = FSDirectory.open(Paths.get("/home/mchenyu/22summer/Integration3/index1/documents.index"));
        // String querystr = keywords.length() > 0 ? keywords : "Economic";
        if(query.length() == 0) query = "machine learning";
  
        MultiFieldQueryParser m = new MultiFieldQueryParser(new String[]{"title", "abst"}, analyzer);

        // int hitsPerPage = number.length() > 0 ? Integer.parseInt(number) : 50;
        // int hitsPerPage = 50; 
        // searchMode = searchMode.length() > 0 ? searchMode : "BM-25";
        

        IndexReader reader = DirectoryReader.open(index);
        IndexSearcher searcher = new IndexSearcher(reader);

        //set similarity with BM-25 or TF-IDF
        if(Objects.equals(searchMode, "BM-25")){
            System.out.println("Searching mode is BM25");
            searcher.setSimilarity(new BM25Similarity(1.0f,0.65f));
        }else{
            System.out.println("Searching mode is TF-IDF");
            
        }


        TopDocs docs = searcher.search(m.parse(query), hitsPerPage);
        ScoreDoc[] hits = docs.scoreDocs;
        System.out.println("Total hits number is " + hits.length);
        // 4. display results
        System.out.println("Found " + hits.length + " hits.");
        List<String> list = new ArrayList<>(); 

        for (int i = 0; i < hits.length; ++i) {
            int docId = hits[i].doc;
            Document d = searcher.doc(docId);
            list.add(d.get("id")); 
            System.out.println((i + 1) + ". " + d.get("id"));

            // + "\t" + d.get("title") + "\t" + d.get("abstract")

            // This line is for getting the score of each document
            // System.out.println(hits[i].score);
        }
        reader.close();
        return list; 
    }


    private static void addDoc(IndexWriter w, String id, String title, String abst) throws IOException {
        Document doc = new Document();
        doc.add(new TextField("id", id, Store.YES));

        doc.add(new TextField("title", title, Store.YES));

        // use a string field for isbn because we don't want it tokenized
        //internation standard book number
        doc.add(new StringField("abstract", abst, Store.YES));
        w.addDocument(doc);
    }




    public static void main(String[] args) throws IOException, ParseException, org.apache.lucene.queryparser.classic.ParseException {

        // try {
        //     File myObj = new File("/home/mchenyu/22summer/Integration3/src/test.txt");
        //     Scanner myReader = new Scanner(myObj);
        //     while (myReader.hasNextLine()) {
        //         String data = myReader.nextLine();
        //         String[] temp = data.split("\\s");
        //         for(int i = 0; i < testData.length; i++)
        //         {
        //             testData[i] = Float.parseFloat(temp[i]);
        //         }
        //     }
        //     myReader.close();
        // } catch (FileNotFoundException e) {
        //     System.out.println("An error occurred.");
        //     e.printStackTrace();
        // }

        setupIndexDir();
        // init();
        // testWriteAndQueryIndex(testData);
       GatewayServer gatewayServer = new GatewayServer(new App());
       gatewayServer.start();
        System.out.println("Gateway Server Started");
    }

    public static void setupIndexDir() throws IOException {
        File file = new File(pathOfCSV);
        int count = 0;
        BufferedReader br = new BufferedReader(new FileReader(file));
        String nextLine;
        while ((nextLine = br.readLine()) != null) {
            String[] cols = nextLine.split("\t");
            String id = Integer.valueOf(count).toString();
            String title = cols[1];
            count++;
            // if(count == 5000) break;
            // System.out.println(title);

            String abst = cols[5];
            // System.out.println(abst);
            String vector50Col = cols[12];

            if( vector50Col.equals("vector_bert")) continue;

            vector50Col = vector50Col.replaceAll("[\\p{Ps}\\p{Pe}]", "");
            strToList(vector50Col, vector50);
        }
        br.close();
        File file1 = indexPath.toFile();
        if (file1.exists()) {
            FileUtils.deleteDirectory(file1);
        }
        // Create the provider which will feed the vectors for the graph
        
//        vectors = new newVectorProvider(vector50);

    }

    static class newVectorProvider extends VectorValues implements RandomAccessVectorValues, RandomAccessVectorValuesProducer{
        int doc = -1;
        private final List<Vector<Float>> data;

        public newVectorProvider(List<Vector<Float>> data) {
            this.data = data;
        }

        public Vector<Float> get(int idx) {
            return data.get(idx);
        }

        @Override
        public RandomAccessVectorValues randomAccess() {
            return new newVectorProvider(data);
        }

        @Override
        public float[] vectorValue(int ord) throws IOException {
            Vector<Float> entry = data.get(ord);
            float[] f = new float[768];
            for(int i = 0; i < 768; i++)
            {
                f[i] = entry.get(i);
            }
            return f;
        }

        @Override
        public BytesRef binaryValue(int targetOrd) throws IOException {
            return null;
        }

        @Override
        public int dimension() {
            return 768;
        }

        @Override
        public int size() {
            return data.size();
        }

        @Override
        public float[] vectorValue() throws IOException {
            return vectorValue(doc);
        }

        @Override
        public int docID() {
            return doc;
        }

        @Override
        public int nextDoc() throws IOException {
            return advance(doc + 1);
        }

        @Override
        public int advance(int target) throws IOException {
            if (target >= 0 && target < data.size()) {
                doc = target;
            } else {
                doc = NO_MORE_DOCS;
            }
            return doc;
        }

        @Override
        public long cost() {
            return data.size();
        }

        public newVectorProvider copy() {
            return new newVectorProvider(data);
        }
    }


    public static List<Integer> testWriteAndQueryIndex(String target) throws IOException {
        // Persist and read the data
        try (MMapDirectory dir = new MMapDirectory(indexPath)) {
            
            vectors = new newVectorProvider(vector50);
            // Write index
            int indexedDoc = writeIndex(dir, vectors);

            // Read index
            return readAndQuery(dir, vectors, indexedDoc, target);
        }
    }

    private static List<Integer> readAndQuery(MMapDirectory dir, newVectorProvider vectorData, int indexedDoc, String target) throws IOException {
            List<Integer> Finalresult = new ArrayList<>();  
            float[] ans = new float[768]; 
            String[] temp = target.split("\\s"); 
            for(int i = 0; i < temp.length; i++)
            {
                ans[i] = Float.parseFloat(temp[i]); 
            }
            
        try (IndexReader reader = DirectoryReader.open(dir)) {
            for (LeafReaderContext ctx : reader.leaves()) {
                System.out.println("This is just a test for the knn");
                VectorValues values = ctx.reader().getVectorValues("field");
                // KnnGraphValues graphValues = ((Lucene90HnswVectorsReader) ((PerFieldKnnVectorsFormat.FieldsReader) ((CodecReader) ctx.reader())
                // .getVectorReader())
                // .getFieldReader("field"))
                // .getGraphValues("field");

                TopDocs results = doKnnSearch(ctx.reader(), "field", ans, 10, indexedDoc);
                System.out.println("Doc Based Search:");
                // System.out.println(Arrays.toString(ans));
                System.out.println("TotalHits: " + results.totalHits.value);
                for (int i = 0; i < results.scoreDocs.length; i++) {
                    ScoreDoc doc = results.scoreDocs[i];
                    Finalresult.add(doc.doc); 
                    System.out.println("Matches: " + doc.doc + " = " + doc.score);
                    // Vector<Float> vec = vectorData.get(doc.doc);

                    // System.out.println(vec.toString());
                }
            }
        }
        return Finalresult; 

    }


    private static int writeIndex(MMapDirectory dir, newVectorProvider vectorProvider) throws IOException {
        int indexedDoc = 0;
        IndexWriterConfig iwc = new IndexWriterConfig()
                .setCodec(
                        new Lucene90Codec() {
                            @Override
                            public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
                                return new Lucene90HnswVectorsFormat(maxConn, beamWidth);
                            }
                        });
        try (IndexWriter iw = new IndexWriter(dir, iwc)) {
            while (vectorProvider.nextDoc() != NO_MORE_DOCS) {
                while (indexedDoc < vectorProvider.docID()) {
                    // increment docId in the index by adding empty documents
                    iw.addDocument(new Document());
                    indexedDoc++;
                }
                Document doc = new Document();
                // System.out.println("Got: " + v2.vectorValue()[0] + ":" + v2.vectorValue()[1] + "@" + v2.docID());
                doc.add(new KnnVectorField("field", vectorProvider.vectorValue(), similarityFunction));
                doc.add(new StoredField("id", vectorProvider.docID()));
                iw.addDocument(doc);
                indexedDoc++;
            }
        }
        return indexedDoc;
    }

    private static TopDocs doKnnSearch(
            IndexReader reader, String field, float[] vector, int docLimit, int fanout) throws IOException {
        TopDocs[] results = new TopDocs[reader.leaves().size()];
        for (LeafReaderContext ctx : reader.leaves()) {
            Bits liveDocs = ctx.reader().getLiveDocs();
            results[ctx.ord] = ctx.reader().searchNearestVectors(field, vector, docLimit + fanout, liveDocs);
            int docBase = ctx.docBase;
            for (ScoreDoc scoreDoc : results[ctx.ord].scoreDocs) {
                scoreDoc.doc += docBase;
            }
        }
        return TopDocs.merge(docLimit, results);
    }

    private static void strToList(String vector100Col, List<Vector<Float>> vector100) {
        String[] vector100Token = vector100Col.split(",");
        Vector<Float> temp1 = new Vector<>();
        for (String ss : vector100Token) {
            float f1 = Float.parseFloat(ss);
            temp1.add(f1);
        }
        vector100.add(temp1);
    }








}
