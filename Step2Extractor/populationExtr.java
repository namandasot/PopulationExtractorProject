/*
 */

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.StringReader;

import edu.stanford.nlp.process.Tokenizer;
import edu.stanford.nlp.process.TokenizerFactory;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.DocumentPreprocessor;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.Sentence;
import edu.stanford.nlp.trees.*;
import edu.stanford.nlp.trees.tregex.TregexMatcher;
import edu.stanford.nlp.trees.tregex.TregexPattern;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import java.util.Scanner;
class populationExtr {

	/**
	 * The main method demonstrates the easiest way to load a parser.
	 * Simply call loadModel and specify the path of a serialized grammar
	 * model, which can be a file, a resource on the classpath, or even a URL.
	 * For example, this demonstrates loading from the models jar file, which
	 * you therefore need to include in the classpath for ParserDemo to work.
	 */
	;
	// Ffw = new FileWriter ("DataSet/New/outNegative1_parse.txt"); ;
	public static void main(String[] args) {
		LexicalizedParser lp = LexicalizedParser.loadModel("edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz");
		Scanner in = new Scanner(System.in);
		int a =0;int totalT =0 ,totalF=0,TP=0,TN=0,FP=0,FN=0;

		File inFile = new File ("DataSet/testClass0.txt");        //input file
		FileWriter fw =null;
		try {
			fw = new FileWriter("DataSet/ResultFile.txt");    //output file

		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		BufferedWriter bw = new BufferedWriter(fw);
		Scanner sc = null;
		File inFile2 = new File ("DataSet/out1.txt");       //Output of Naive-Bayes Classifier file


		int k =1;// k=1 implies We should consider Population abstract found as positive event. 
		//k=0 implies population abstract not found as positive event.


		Scanner ansFile =null;

		try {
			sc = new Scanner (inFile);
			ansFile  =  new Scanner (inFile2);

		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		String sent3 = ansFile.nextLine();
		while(sc.hasNextLine())
		{

			a++;
			int turn =0;
			int done =0;String tregexPattern ="";
			String T = "";
			String sent2 = sc.nextLine();
			sent2 = sent2.toLowerCase();
			String[] temp4;
			temp4 = sent3.split(" ");
			System.out.println(temp4[0]);
			int x = Integer.parseInt(temp4[0]);

			if(x==1)                                             
			{
				Tree parse = demoAPI(lp,sent2);



				while(done == 0)
				{
					if(turn==0)
						tregexPattern = "NP > PP";
					else if(turn==1)
						tregexPattern = "PP , PN";
					else if(turn==2)
						tregexPattern = "PP $ PN";
					else if(turn==3)
						tregexPattern = "PP $ PP";
					else if (turn==4)
						tregexPattern = "NP $ NP ";
					else if (turn==5)
						tregexPattern = "NP $ NN";
					else if (turn ==6)
						tregexPattern = "NP , PP";
					else if (turn ==7)
						tregexPattern = "NP $ PP";
					else if (turn ==8)
						tregexPattern = "@NP";
					else if (turn ==9)
						tregexPattern = "@VP";
					else
						break;
					TregexPattern p = TregexPattern.compile(tregexPattern); //@NP targets NP in the tree
					TregexMatcher m	= p.matcher(parse);
					String S ="";			
					if(turn ==0 || turn ==1 )
					{
						while(m.findNextMatchingNode()) {
							if(m.getMatch().toString().contains("patient"))//||m.getMatch().toString().contains("with"))
								S= m.getMatch().toString();
						}
					}
					else
					{
						while(m.findNextMatchingNode()) {
							if(m.getMatch().toString().contains("patient")||m.getMatch().toString().contains("with"))
								S = S.concat(m.getMatch().toString());
						}
					}
					T = "";String temp;

					for(int i=0;i<S.length();i++)
					{
						temp = Character.toString(S.charAt(i));
						if(S.charAt(i) >=97 && S.charAt(i) <= 122 || (S.charAt(i)==32 ))
						{
							T = T.concat(temp);    		  
						}
					}
					T = T.trim();
					String[] temp1;
					T = T.replaceAll("  ", ",");
					temp1 = T.split("[,]");
					if(temp1.length >2)
					{
						if(temp1[0].toString().contains("patient")|| temp1[1].contains("patient")||temp1[2].contains("patient"))
						{
							done = 1;//System.out.println("Done Changed to 1");
						}
					}
					if(temp1[temp1.length-1].toString().contains("patient"))
					{
						//done = 0;//System.out.println("Done Changed to 1");
					}
					if(done ==0)
					{turn++;}//System.out.println("turn++");}
				}
				if(done ==1 )
				{ T = T.replaceAll(",", " ");System.out.println(a + " " + turn + " Found Match :"+ T + " " +totalT + " " + totalF);
				if(k==0)
				{totalF++;FP++;}
				else {totalT++;TP++;}
				//					} catch (IOException e) {
				// TODO Auto-generated catch block
				//						e.printStackTrace();
				//					}
				}
				else 
				{
					System.out.println(a +" " + sent2 + "\n  Not Found "+totalT + " " + totalF);
					try {
						bw.write(a + " " + sent2 + "\n*Not Found*" +"\n\n"); 
						if(k==0)
						{totalT++;TN++;}
						else {totalF++;FN++;}
					} catch (IOException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				}
			}
			else
			{


				System.out.println(a + " Sec Not Found " +totalT + " " + totalF);
				try {
					bw.write(a + " " + sent2 + "\n*Not Found*" +"\n\n"); 
					if(k==0)
					{totalT++;TN++;}
					else {totalF++;FN++;}
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}

		}
		System.out.println("\nTotal True: " + totalT + "\ntotal False : " + totalF);
		System.out.println("TP "+TP + " TN " + TN + " FP " + FP + " FN " +FN);
		try {
			bw.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	/**
	 * demoDP demonstrates turning a file into tokens and then parse
	 * trees.  Note that the trees are printed by calling pennPrint on
	 * the Tree object.  It is also possible to pass a PrintWriter to
	 * pennPrint if you want to capture the output.
	 */

	public static Tree  demoAPI(LexicalizedParser lp,String sent2) {
		// This option shows parsing a list of correctly tokenized words



		// This option shows loading and using an explicit tokenizer

		TokenizerFactory<CoreLabel> tokenizerFactory =
				PTBTokenizer.factory(new CoreLabelTokenFactory(), "");
		Tokenizer<CoreLabel> tok =
				tokenizerFactory.getTokenizer(new StringReader(sent2));
		List<CoreLabel> rawWords2 = tok.tokenize();
		Tree parse = lp.apply(rawWords2);

		TreebankLanguagePack tlp = new PennTreebankLanguagePack();
		GrammaticalStructureFactory gsf = tlp.grammaticalStructureFactory();
		GrammaticalStructure gs = gsf.newGrammaticalStructure(parse);
		Collection<TypedDependency> tdl = gs.typedDependencies();
		//System.out.println(tdl);
		//System.out.println("TP WALA\n");

		// You can also use a TreePrint object to print trees and dependencies
		TreePrint tp = new TreePrint("penn,typedDependencies");
		tp.printTree(parse);
		return parse;
	}

	private populationExtr() {} // static methods only

}
