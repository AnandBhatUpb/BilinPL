package tests.examples;

import disc.bilinpl.BilinPLTrainer;
import disc.utilities.Evaluation;
import disc.utilities.MyUtils;
import disc.utilities.bilinpl.BilinPLUtils;
import disc.utilities.bilinpl.KronUtils;
import disc.utilities.bilinpl.Permutation;
import py4j.GatewayServer;

public class JavaGateWay {
	public static void trainAndTestBilinPL(double [][] labelFeatures, double [][] trInstances, double [][] teInstances, int [][] trOrderings, double [][] d_teRankings) {
		// Kronecker Product
		double [][][] jf = KronUtils.getOrderedKronJointFeatureTensor(trInstances, labelFeatures, trOrderings);
		// --
		BilinPLTrainer trainer = new BilinPLTrainer();
		trainer.init(jf);
		long millis_start = System.currentTimeMillis();// % 1000;
		double[] theta_prime = trainer.train();
		long millis_end = System.currentTimeMillis();// % 1000;
		System.out.println("Took "+(millis_end-millis_start)+" ms");
		
		System.out.println("Solution:");
		MyUtils.printDoubleArray(theta_prime);
		
		// Predictions
		double [][][] jf_test = KronUtils.getKronJointFeatureTensor(teInstances, labelFeatures);
		double [][] pred_orderings =  trainer.predictSet(jf_test, theta_prime);
		
		
		
		// Evaluate
		double ktau = 0d;
		for (int i=0;i<d_teRankings.length;i++) {
			double [] pred_ranking = Permutation.switchOrderingRanking(pred_orderings[i]);
			ktau = ktau + Evaluation.kendall(d_teRankings[i], pred_ranking);
		}
		ktau = ktau/(double)d_teRankings.length;
		System.out.println("Ktau Test : "+ktau);
	}
	
	public static void trainAndTestLinPL(double [][] labelFeatures, double [][] trInstances, double [][] teInstances, int [][] trOrderings, double [][] d_teRankings) {
		//Step 1: 1d 0d array of length [7][7]
		double [][] linplLabelFeatures = BilinPLUtils.constructFeaturesForLinPL(labelFeatures.length);
		System.out.println("Step 1 output: "+ linplLabelFeatures[0][0]);
		
		//Step 2: 
		double [][][] jf = KronUtils.getOrderedKronJointFeatureTensor(trInstances, linplLabelFeatures, trOrderings);
		// --
		BilinPLTrainer trainer = new BilinPLTrainer();
		trainer.init(jf);
		long millis_start = System.currentTimeMillis();// % 1000;
		double[] theta_prime = trainer.train();
		long millis_end = System.currentTimeMillis();// % 1000;
		System.out.println("Took "+(millis_end-millis_start)+" ms");
		
		System.out.println("Solution:");
		MyUtils.printDoubleArray(theta_prime);
		
		// Predictions
		double [][][] jf_test = KronUtils.getKronJointFeatureTensor(teInstances, linplLabelFeatures);
		double [][] pred_orderings =  trainer.predictSet(jf_test, theta_prime);
		
		// Evaluate
		double ktau = 0d;
		for (int i=0;i<d_teRankings.length;i++) {
			double [] pred_ranking = Permutation.switchOrderingRanking(pred_orderings[i]);
			ktau = ktau + Evaluation.kendall(d_teRankings[i], pred_ranking);
		}
		ktau = ktau/(double)d_teRankings.length;
		System.out.println("Ktau Test : "+ktau);
	}
	
	public static void trainAndTestLinSidePL(double [][] labelFeatures, double [][] trInstances, double [][] teInstances, int [][] trOrderings, double [][] d_teRankings) {
		double [][] linsideplLabelFeatures = BilinPLUtils.constructFeaturesForLinSidePL(labelFeatures);
		
		double [][][] jf = KronUtils.getOrderedKronJointFeatureTensor(trInstances, linsideplLabelFeatures, trOrderings);
		// --
		BilinPLTrainer trainer = new BilinPLTrainer();
		trainer.init(jf);
		long millis_start = System.currentTimeMillis();// % 1000;
		double[] theta_prime = trainer.train();
		long millis_end = System.currentTimeMillis();// % 1000;
		System.out.println("Took "+(millis_end-millis_start)+" ms");
		
		System.out.println("Solution:");
		MyUtils.printDoubleArray(theta_prime);
		
		// Predictions
		double [][][] jf_test = KronUtils.getKronJointFeatureTensor(teInstances, linsideplLabelFeatures);
		double [][] pred_orderings =  trainer.predictSet(jf_test, theta_prime);
		
		// Evaluate
		double ktau = 0d;
		for (int i=0;i<d_teRankings.length;i++) {
			double [] pred_ranking = Permutation.switchOrderingRanking(pred_orderings[i]);
			ktau = ktau + Evaluation.kendall(d_teRankings[i], pred_ranking);
		}
		ktau = ktau/(double)d_teRankings.length;
		System.out.println("Ktau Test : "+ktau);
	}
	
	public static void print(String m) {
		System.out.println(m);
	}

	public static void main(String[] args) throws Exception {
		/*String filename = "data/CPAdataNoZeros.mat";
		MatFileReader mreader = new MatFileReader(filename);
		Map<String,MLArray> map = mreader.getContent();
	
		MLDouble mld;
		mld = (MLDouble) map.get("labelFeatures");
		double [][] labelFeatures = mld.getArray();
		MyUtils.printDoubleArray(labelFeatures[0]);
		
		mld = (MLDouble) map.get("instTrFeatures");
		double [][] trInstances = mld.getArray();
		MyUtils.printDoubleArray(trInstances[0]);
		
		mld = (MLDouble) map.get("instTeFeatures");
		double [][] teInstances = mld.getArray();
		MyUtils.printDoubleArray(teInstances[0]);
		
		mld = (MLDouble) map.get("trOrderings");
		double [][] d_trOrderings = mld.getArray();
		MyUtils.printDoubleArray(d_trOrderings[0]);
		int [][] trOrderings = MyUtils.convertDouble2dArrayToIntArray(d_trOrderings);
		MyUtils.printIntArray(trOrderings[0]);
		
		mld = (MLDouble) map.get("teRankings");
		double [][] d_teRankings = mld.getArray();
		MyUtils.printDoubleArray(d_teRankings[1]);
		int [][] teRankings = MyUtils.convertDouble2dArrayToIntArray(d_teRankings);
		MyUtils.printIntArray(teRankings[0]);*/
        //--
        
		//System.exit(0);
		// --
		// BILIN PL
		//trainAndTestBilinPL(labelFeatures, trInstances,teInstances, trOrderings, d_teRankings);
		// --
		// LIN PL
		//trainAndTestLinPL(labelFeatures, trInstances,teInstances, trOrderings, d_teRankings);
		// --
		// LINSIDE PL
		//trainAndTestLinSidePL(labelFeatures, trInstances,teInstances, trOrderings, d_teRankings);
		
		JavaGateWay ml = new JavaGateWay();
        // app is now the gateway.entry_point
        GatewayServer server = new GatewayServer(ml);
        server.start();
	}
}
