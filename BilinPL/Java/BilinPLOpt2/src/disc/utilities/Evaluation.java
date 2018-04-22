package disc.utilities;

import java.util.ArrayList;


public class Evaluation {
	
	 
	  public static double spearman(double[] a, double[] b) {
		  return 0d;
	  }
	  
	  public static double spearman(ArrayList<Double> a, ArrayList<Double> b) {
		  
		  return 0d;
	  }
	  
//	  public static double kendall(double []a, double[] b) {
//		  int M = a.length;
//		  int C = 0;
//		  int D = 0;
//		  for (int i=0;i<M;i++) {
//			  for (int j=i+1;j<M;j++) {
//				  double p = (a[i]-a[j])*(b[i]-b[j]);
//				  if (p>=0) {
//					  C = C +1;
//				  } else {
//					  D = D + 1;
//				  }
//			  }
//		  }
//		  double denom=(double)M*((double)M-1)/(double)2;
//		  return (double)(C-D)/denom;
//	  }
	  
	  // Ranking measure for partial ranking
	  public static double kendall(double []a, double[] b) {
		  int M = a.length;
		  int i;
		  int C=0;
		  for(i=0; i<M; i++) {
			  if((double)a[i] != 0.0) {
				  System.out.println("entered and value is : "+a[i]);
				  System.out.println("b value is :"+ b[i]);
				  if(a[i] == b[i]) {
					  System.out.println("entered second condition value is :"+b[i]);
					  C = C+1;
				  }
			  }
		  }
		  
		  return (double)C/10d;
	  }

	    
	  public static double kendall(ArrayList<Double> a, ArrayList<Double> b) {
		  double [] da = new double[a.size()];
		  double [] db = new double[a.size()];
		  for (int i=0;i<a.size();i++) {
			  da[i] = a.get(i);
			  db[i] = b.get(i);
		  }
		  return kendall(da,db);
	  }
	  
	  public static double standarddeviation(double [] data) {
		  return StdStats.stddev(data);
		 
	  }
	  
	  
	  public static double footrule(double [] a, double[] b) {
		  double s = 0d;
		  int len = a.length;
		  for (int i=0;i<len;i++) {
			  s = s + Math.abs(a[i]-b[i]);
		  }
		  return s;
	  }
	  public static double footruleDistance(ArrayList<Double> a, ArrayList<Double> b) {
		  double [] da = new double[a.size()];
		  double [] db = new double[a.size()];
		  for (int i=0;i<a.size();i++) {
			  da[i] = a.get(i);
			  db[i] = b.get(i);
		  }
		  return footrule(da,db);
	  }
	  
	  public static double dotProduct(double [] x, double[] y){
		  int l = x.length;
		  double result = 0d;
		  for (int i=0;i<l;i++){
			  result += x[i]*y[i];
		  }
		  return result;
	  }
}
