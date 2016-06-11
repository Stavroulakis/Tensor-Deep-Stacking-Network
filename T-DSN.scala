import scala.util.Random
import scala.math
import java.io.File
import java.util.Arrays
import breeze.stats.distributions._
import breeze.linalg._


def sigmoid_function(x: Double): Double = {
	return 1.0 / (1.0 + math.pow(math.E, -x))
	 }

def output(input: breeze.linalg.DenseVector[Double], w: breeze.linalg.DenseMatrix[Double], bias: Double): breeze.linalg.DenseVector[Double] = {
	
    var w_new=w.t
    var linear_output=DenseVector.zeros[Double](w_new.rows)
    var len1=w.rows
    var len2=w_new.rows
	for(i<- 0 until len2) {
        for(j<-0 until len1){
            linear_output(i) += w_new(i,j) * input(j)
        }
	    linear_output(i) += bias
    }
	
    var h=DenseVector.zeros[Double](w_new.rows)
	for( k <- 0 until w_new.rows) {
        h(k)=sigmoid_function(linear_output(k))  
    }
    return h    
	}

def gauss_weigth_create(mean:Double,dias:Double):Double={
    var  rng = new Random()
    var num = rng.nextGaussian()*dias + mean
    return num
}

def weigth_create(min:Double,max:Double):Double={
 	var  rng = new Random()
	var range=max-min
	var num = rng.nextDouble() * range
	var  fin_value = num+min
	return fin_value
}

def kha_Rao_prod(H1: breeze.linalg.DenseMatrix[Double],H2:breeze.linalg.DenseMatrix[Double]): breeze.linalg.DenseMatrix[Double]={

    var row=H1.rows * H2.rows
    var col=H1.cols
    var H=DenseMatrix.zeros[Double](row,col)
    for(k<-0 until col){
        var a=H2(::,k)
        var t=0
        for(i<-0 until H1.rows){
            for(j<-0 until H1.rows){
                var B=H1(i,k)*a(j)
                H(t,k)=B
                t=t+1
            }
        }

    }
    return H
}

def cross_Matrix(H: breeze.linalg.DenseMatrix[Double]): breeze.linalg.DenseMatrix[Double]={

    var a=H.t
    var b=H*a
    var c=pinv(b)
    return a*c
}

def thita_create(h:breeze.linalg.DenseMatrix[Double],t:breeze.linalg.DenseMatrix[Double],hcross:breeze.linalg.DenseMatrix[Double]):breeze.linalg.DenseMatrix[Double]={

    var a= hcross:*2.0
    var b=h*(t.t)
    var c=t*hcross
    var d=(a*b)*c
    var e=(t.t)*c
    return d-e
}

def psi1_create(h:breeze.linalg.DenseMatrix[Double],thit:breeze.linalg.DenseMatrix[Double]):breeze.linalg.DenseMatrix[Double]={
    var row=h.rows
    var col=h.cols
    var Ψ=DenseMatrix.zeros[Double](row,col)
    for(i<-0 until row){
        for(j<-0 until col){
            var E=DenseMatrix.zeros[Double](row,col)
            E(i,j)=1
            var a=kha_Rao_prod(E,h)
            var b=(thit.t):*a
            Ψ(i,j)=sum(b)
        }
    }
    return Ψ
}

def psi2_create(h:breeze.linalg.DenseMatrix[Double],thit:breeze.linalg.DenseMatrix[Double]):breeze.linalg.DenseMatrix[Double]={
    var row=h.rows
    var col=h.cols
    var Ψ=DenseMatrix.zeros[Double](row,col)
    for(i<-0 until row){
        for(j<-0 until col){
            var E=DenseMatrix.zeros[Double](row,col)
            E(i,j)=1
            var a=kha_Rao_prod(h,E)
            var b=(thit.t):*a
            Ψ(i,j)=sum(b)

        }
    }
    return Ψ
}

def Weigth_update(x:breeze.linalg.DenseMatrix[Double],h:breeze.linalg.DenseMatrix[Double],psi:breeze.linalg.DenseMatrix[Double],w:breeze.linalg.DenseMatrix[Double]):breeze.linalg.DenseMatrix[Double]={

  var a=1.0:-(h.t)
  var b=(h.t):*a
  var c= b:*psi
  var d=x*c
  var e=0.001*d
  return w + e 

}

def MSE(x:breeze.linalg.DenseMatrix[Double],y:breeze.linalg.DenseMatrix[Double]):Double={
    var sum =0.0
    var temp=0.0
    for(i<-0 until x.rows){
        for(j<-0 until x.cols){
           temp=x(i,j)-y(i,j)  
           
           sum=sum + math.pow(temp,2)
        }
    }
    return (sum/(x.cols))*100
}


   def training_network(X:breeze.linalg.DenseMatrix[Double],T:breeze.linalg.DenseMatrix[Double],error:Double,Bias:Double,iter:Double,wlength:Int,a1:breeze.linalg.DenseMatrix[Double],a2:breeze.linalg.DenseMatrix[Double],a3:breeze.linalg.DenseMatrix[Double],a4:breeze.linalg.DenseMatrix[Double],a5:breeze.linalg.DenseMatrix[Double],a6:breeze.linalg.DenseMatrix[Double],a7:breeze.linalg.DenseMatrix[Double],a8:breeze.linalg.DenseMatrix[Double],reg:Int)={
    /////////////////////////////////////////////////////////////////////////
    /////////////////////////FIRST LAYER///////////////////////////////////
    ///////////////////////////////////////////////////////////////////////    
        var ytest=0
        var mse=DenseMatrix.zeros[Double](30,1)
        var mse2=DenseMatrix.zeros[Double](30,1)
        var mse3=DenseMatrix.zeros[Double](30,1)
        var mse4=DenseMatrix.zeros[Double](30,1)
        var counter1=0
        var counter2=0
        var counter3=0
        var counter4=0  
        var W1=a1
        var W2=a2
        var W3=a3
        var W4=a4
        var W5=a5
        var W6=a6
        var W7=a7
        var W8=a8
    
        if(W1(0,0)==0.0 & W1(0,1)==0.0 & W1(1,1)==0){
            for(i<-0 until W1.rows){
        	   for(j<-0 until W1.cols){
        		   W1(i,j)=weigth_create(((-2.4)/X.rows)/2,(2.4/X.rows)/2)
        	   }
            }
        }
      
        if(W2(0,0)==0.0 & W2(0,1)==0.0 & W2(1,1)==0){
            for(i<-0 until W2.rows){
        	   for(j<-0 until W2.cols){
        		  W2(i,j)=weigth_create(((-2.4)/X.rows)/2,(2.4/X.rows)/2) 
        	   }
            }
        }
        var H1=DenseMatrix.zeros[Double](W1.cols,X.cols)
        var H2=DenseMatrix.zeros[Double](W1.cols,X.cols)

        for(i<-0 until X.cols){

        	var temp=X(::,i)
        	//H1 of hidden	
        	var h1=output(temp,W1,Bias)
        	for(j<-0 until h1.length){
        		H1(j,i)=h1(j)
        	}
        	
        	//H2 of hidden	
        	var h2=output(temp,W2,Bias)
        	for(j<-0 until h2.length){
        		H2(j,i)=h2(j)
        	}
        }

        var H=kha_Rao_prod(H1,H2)
        var Hcross=cross_Matrix(H)
        var Utran=T*Hcross
        var Y=Utran*H
        var Θ=thita_create(H,T,Hcross)

        //variables inside the loop
        var Hcross1=DenseMatrix.zeros[Double](Hcross.rows,Hcross.cols)

        //loop for Weigth matrix update
        do{
            counter1+=1
            var Ψ1=psi1_create(H2,Θ)
            var Ψ2=psi2_create(H1,Θ)

            W1=Weigth_update(X,H1,Ψ1,W1)    
            W2=Weigth_update(X,H2,Ψ2,W2)
            var H11=DenseMatrix.zeros[Double](W1.cols,X.cols)
            var H22=DenseMatrix.zeros[Double](W2.cols,X.cols)

            for(i<-0 until X.cols){
                var temp=X(::,i)
                //H1 of hidden  
                var h1=output(temp,W1,Bias)
                for(j<-0 until h1.length){
                    H11(j,i)=h1(j)
                }
            
                 //H2 of hidden    
                var h2=output(temp,W2,Bias)
                for(j<-0 until h2.length){
                    H22(j,i)=h2(j)
                }
            }
            H=kha_Rao_prod(H11,H22)
            Hcross1=cross_Matrix(H)
            var Θ1=thita_create(H,T,Hcross1)
            Θ=Θ1
            H1=H11
            H2=H22
            /////check Mse to break loop
            var utest=T*Hcross1
            var ytest=utest*H
            
            ////regulize
            if(reg==1){
                for(i<-0 until ytest.cols){
                    if(ytest(0,i)>0.5){
                        ytest(0,i)=1.0
                    }
                     else if(ytest(0,i)<(-0.5)){
                        ytest(0,i)=(-1.0)
                    }
                    else{
                        ytest(0,i)=0.0
                    }
                }
            }                
             mse(counter1-1,0) =MSE(ytest,T)
             println(mse(counter1 -1,0))
        }while( error < mse(counter1 -1,0) || counter1 < iter)   
       
        //output of first Layer
        var Utran1=T*Hcross1
        var Y1=Utran1*H
        if(reg==1){
                for(i<-0 until Y1.cols){
                    if(Y1(0,i)>0.5){
                        Y1(0,i)=1.0
                    }
                     else if(Y1(0,i)<(-0.5)){
                        Y1(0,i)=(-1.0)
                    }
                    else{
                        Y1(0,i)=0.0
                    }
                }
            }  

       println("Finish 1 layer")
    /////////////////////////////////////////////////////////////////////////
    /////////////////////////SECOND LAYER///////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
        
        var Xnew=DenseMatrix.vertcat(X,Y1)
        if(W3(0,0)==0.0 & W3(0,1)==0.0 & W3(1,1)==0.0 ){
            for(i<-0 until W3.rows){
                for(j<-0 until W3.cols){
                     W3(i,j)=weigth_create(((-2.4)/Xnew.rows)/2,(2.4/Xnew.rows)/2)
                }
            }
        }
       
        if( W4(0,0)==0.0 & W4(0,1)==0.0 & W4(1,1)==0.0 ){ 
            for(i<-0 until W4.rows){
                for(j<-0 until W4.cols){
                     W4(i,j)=weigth_create(((-2.4)/Xnew.rows)/2,(2.4/Xnew.rows)/2)
                }
            }
        }
        var H3=DenseMatrix.zeros[Double](W3.cols,Xnew.cols)
        var H4=DenseMatrix.zeros[Double](W4.cols,Xnew.cols)

        for(i<-0 until Xnew.cols){

            var temp=Xnew(::,i)
            //H3 of hidden  
            var h3=output(temp,W3,Bias)
            for(j<-0 until h3.length){
                H3(j,i)=h3(j)
            }
            
            //H4 of hidden    
            var h4=output(temp,W4,Bias)
            for(j<-0 until h4.length){
                H4(j,i)=h4(j)
            }
        }

         H=kha_Rao_prod(H3,H4)
         Hcross=cross_Matrix(H)
         Utran=T*Hcross
         Y=Utran*H

         Θ=thita_create(H,T,Hcross)

        //variables inside the loop
        var Hcross2=DenseMatrix.zeros[Double](Hcross.rows,Hcross.cols)

        //loop for Weigth matrix update
        do{
            counter2+=1
            var Ψ3=psi1_create(H4,Θ)
            var Ψ4=psi2_create(H3,Θ)
            W3=Weigth_update(Xnew,H3,Ψ3,W3)    
            W4=Weigth_update(Xnew,H4,Ψ4,W4)
            var H33=DenseMatrix.zeros[Double](W3.cols,Xnew.cols)
            var H44=DenseMatrix.zeros[Double](W4.cols,Xnew.cols)
            for(i<-0 until Xnew.cols){

                var temp=Xnew(::,i)
                //H3 of hidden  
                var h3=output(temp,W3,Bias)
                for(j<-0 until h3.length){
                    H33(j,i)=h3(j)
                }
            
                 //H4 of hidden    
                var h4=output(temp,W4,Bias)
                for(j<-0 until h4.length){
                    H44(j,i)=h4(j)
                }
            }

            H=kha_Rao_prod(H33,H44)
            Hcross2=cross_Matrix(H)
            var Θ1=thita_create(H,T,Hcross2)
            Θ=Θ1
            H3=H33
            H4=H44

            /////check Mse to break loop
            var utest2=T*Hcross2
            var ytest2=utest2*H
            
            //regulate
            if(reg==1){
                for(i<-0 until ytest2.cols){
                    if(ytest2(0,i)>0.5){
                        ytest2(0,i)=1.0
                    }
                    else if(ytest2(0,i)<(-0.5)){
                        ytest2(0,i)=(-1.0)
                    }
                    else{
                        ytest2(0,i)=0.0
                    }
                }
            }    

             mse2(counter2 -1,0) =MSE(ytest2,T)
             println(mse2(counter2 -1,0))
        }while(error < mse2(counter2 -1,0) || counter2 < iter )   

        //output of second Layer
        var Utran2=T*Hcross2
        var Y2=Utran2*H

        if(reg==1){
                for(i<-0 until Y2.cols){
                    if(Y2(0,i)>0.5){
                        Y2(0,i)=1.0
                    }
                    else if(Y2(0,i)<(-0.5)){
                        Y2(0,i)=(-1.0)
                    }
                    else{
                        Y2(0,i)=0.0
                    }
                }
            }

        println("Finish 2 layer")
    /////////////////////////////////////////////////////////////////////////
    /////////////////////////THIRD LAYER///////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
        var Xnew2=DenseMatrix.vertcat(X,Y2)
        if(W5(0,0)==0.0 & W5(0,1)==0.0 & W5(1,1)==0.0){
            for(i<-0 until W5.rows){
                for(j<-0 until W5.cols){
                     W5(i,j)=weigth_create(((-2.4)/Xnew2.rows)/2,(2.4/Xnew2.rows)/2)
                }
            }
        }
        if(W6(0,0)==0.0 & W6(0,1)==0.0 & W6(1,1)==0.0){
            for(i<-0 until W6.rows){
                for(j<-0 until W6.cols){
                     W6(i,j)=weigth_create(((-2.4)/Xnew2.rows)/2,(2.4/Xnew2.rows)/2)
                }
            }
        }

        var H5=DenseMatrix.zeros[Double](W5.cols,Xnew2.cols)
        var H6=DenseMatrix.zeros[Double](W6.cols,Xnew2.cols)

        for(i<-0 until Xnew2.cols){

            var temp=Xnew2(::,i)
            //H5 of hidden  
            var h5=output(temp,W5,Bias)
            for(j<-0 until h5.length){
                H5(j,i)=h5(j)
            }
            
            //H6 of hidden    
            var h6=output(temp,W6,Bias)
            for(j<-0 until h6.length){
                H6(j,i)=h6(j)
            }
        }

         H=kha_Rao_prod(H5,H6)
         Hcross=cross_Matrix(H)
         Utran=T*Hcross
         Y=Utran*H

         Θ=thita_create(H,T,Hcross)

        //variables inside the loop
        var Hcross3=DenseMatrix.zeros[Double](Hcross.rows,Hcross.cols)

        //loop for Weigth matrix update
        do{
            counter3+=1
            var Ψ5=psi1_create(H6,Θ)
            var Ψ6=psi2_create(H5,Θ)

            W5=Weigth_update(Xnew2,H5,Ψ5,W5)    
            W6=Weigth_update(Xnew2,H6,Ψ6,W6)
            var H55=DenseMatrix.zeros[Double](W5.cols,Xnew2.cols)
            var H66=DenseMatrix.zeros[Double](W6.cols,Xnew2.cols)

            for(i<-0 until Xnew2.cols){

                var temp=Xnew2(::,i)
                //H5 of hidden  
                var h5=output(temp,W5,Bias)
                for(j<-0 until h5.length){
                    H55(j,i)=h5(j)
                }
            
                 //H6 of hidden    
                var h6=output(temp,W6,Bias)
                for(j<-0 until h6.length){
                    H66(j,i)=h6(j)
                }
            }

            H=kha_Rao_prod(H55,H66)
            Hcross3=cross_Matrix(H)
            var Θ1=thita_create(H,T,Hcross3)
            Θ=Θ1
            H5=H55
            H6=H66

         /////check Mse to break loop
            var utest3=T*Hcross3
            var ytest3=utest3*H
             
             //regulate
            if(reg==1){
                for(i<-0 until ytest3.cols){
                    if(ytest3(0,i)>0.5){
                        ytest3(0,i)=1.0
                    }
                    else if(ytest3(0,i)<(-0.5)){
                        ytest3(0,i)=(-1.0)
                    }
                    else{
                        ytest3(0,i)=0.0
                    }
                }
            }  

             mse3(counter3 -1,0) =MSE(ytest3,T)
             println(mse3(counter3 -1,0))
        }while(error < mse3(counter3 -1,0) || counter3 < iter  )    

        //output of third Layer
        var Utran3=T*Hcross3
        var Y3=Utran3*H

         if(reg==1){
                for(i<-0 until Y3.cols){
                    if(Y3(0,i)>0.5){
                        Y3(0,i)=1.0
                    }
                    else if(Y3(0,i)<(-0.5)){
                        Y3(0,i)=(-1.0)
                    }
                    else{
                        Y3(0,i)=0.0
                    }
                }
            }  
        
        println("Finish 3 layer")
    /////////////////////////////////////////////////////////////////////////
    /////////////////////////FOURTH LAYER///////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
        var Xnew3=DenseMatrix.vertcat(X,Y3)
        if(W7(0,0)==0.0 & W7(0,1)==0.0 & W7(1,1)==0.0){
            for(i<-0 until W7.rows){
                for(j<-0 until W7.cols){
                     W7(i,j)=weigth_create(((-2.4)/Xnew3.rows)/2,(2.4/Xnew3.rows)/2)
                }
            }
        }
       
        if(W8(0,0)==0.0 & W8(0,1)==0.0 & W8(1,1)==0.0){
            for(i<-0 until W8.rows){
                for(j<-0 until W8.cols){
                    W8(i,j)=weigth_create(((-2.4)/Xnew3.rows)/2,(2.4/Xnew3.rows)/2)
                }
            }
        }
        var H7=DenseMatrix.zeros[Double](W7.cols,Xnew3.cols)
        var H8=DenseMatrix.zeros[Double](W8.cols,Xnew3.cols)

        for(i<-0 until Xnew3.cols){

            var temp=Xnew3(::,i)
            //H7 of hidden  
            var h7=output(temp,W7,Bias)
            for(j<-0 until h7.length){
                H7(j,i)=h7(j)
            }
            
            //H8 of hidden    
            var h8=output(temp,W8,Bias)
            for(j<-0 until h8.length){
                H8(j,i)=h8(j)
            }
        }

         H=kha_Rao_prod(H7,H8)
         Hcross=cross_Matrix(H)
         Utran=T*Hcross
         Y=Utran*H

         Θ=thita_create(H,T,Hcross)

        //variables inside the loop
        var Hcross4=DenseMatrix.zeros[Double](Hcross.rows,Hcross.cols)

        //loop for Weigth matrix update
        do{
            counter4+=1
            var Ψ7=psi1_create(H8,Θ)
            var Ψ8=psi2_create(H7,Θ)

            W7=Weigth_update(Xnew3,H7,Ψ7,W7)    
            W8=Weigth_update(Xnew3,H8,Ψ8,W8)
            var H77=DenseMatrix.zeros[Double](W7.cols,Xnew3.cols)
            var H88=DenseMatrix.zeros[Double](W8.cols,Xnew3.cols)

            for(i<-0 until Xnew3.cols){

                var temp=Xnew3(::,i)
                //H7 of hidden  
                var h7=output(temp,W7,Bias)
                for(j<-0 until h7.length){
                    H77(j,i)=h7(j)
                }
            
                //H8 of hidden    
                var h8=output(temp,W8,Bias)
                for(j<-0 until h8.length){
                    H88(j,i)=h8(j)
                }
            }

            H=kha_Rao_prod(H77,H88)
            Hcross4=cross_Matrix(H)
            var Θ1=thita_create(H,T,Hcross4)
            Θ=Θ1
            H7=H77
            H8=H88

         /////check Mse to break loop
            var utest4=T*Hcross4
            var ytest4=utest4*H
            
             //regulate
            if(reg==1){
                for(i<-0 until ytest4.cols){
                    if(ytest4(0,i)>0.5){
                        ytest4(0,i)=1.0
                    }
                    else if(ytest4(0,i)<(-0.5)){
                        ytest4(0,i)=(-1.0)
                    }
                    else{
                        ytest4(0,i)=0.0
                    }
                }
            }  

             mse4(counter4 -1,0) =MSE(ytest4,T)
             println(mse4(counter4 -1,0))
        }while(error < mse4(counter4 -1,0)  || counter4 < iter )   

        //output of fourth Layer
        var Utran4=T*Hcross4
        var Y4=Utran4*H

        if(reg==1){
                for(i<-0 until Y4.cols){
                    if(Y4(0,i)>0.5){
                        Y4(0,i)=1.0
                    }
                    else if(Y4(0,i)<(-0.5)){
                        Y4(0,i)=(-1.0)
                    }
                    else{
                        Y4(0,i)=0.0
                    }
                }
            }  
 
        println("Finish 4 layer")
        //return all the useful info after training(weigth matrix etc)
        (W1,W2,Utran1,Y1,W3,W4,Utran2,Y2,W5,W6,Utran3,Y3,W7,W8,Utran4,Y4,mse,mse2,mse3,mse4)
    }

    def predict(X:breeze.linalg.DenseMatrix[Double],W1:breeze.linalg.DenseMatrix[Double],W2:breeze.linalg.DenseMatrix[Double],U1:breeze.linalg.DenseMatrix[Double],W3:breeze.linalg.DenseMatrix[Double],W4:breeze.linalg.DenseMatrix[Double],U2:breeze.linalg.DenseMatrix[Double],W5:breeze.linalg.DenseMatrix[Double],W6:breeze.linalg.DenseMatrix[Double],U3:breeze.linalg.DenseMatrix[Double],W7:breeze.linalg.DenseMatrix[Double],W8:breeze.linalg.DenseMatrix[Double],U4:breeze.linalg.DenseMatrix[Double],Bias:Double)={

        /////////////////////////////////////////////////////////////////////////
        /////////////////////////FIRST LAYER///////////////////////////////////
        /////////////////////////////////////////////////////////////////////// 

        var input=X
        var H1=DenseMatrix.zeros[Double](W1.cols,input.cols)
        var H2=DenseMatrix.zeros[Double](W1.cols,input.cols)

        for(i<-0 until input.cols){

            var temp=input(::,i)
            //H1 of hidden  
            var h1=output(temp,W1,Bias)
            for(j<-0 until h1.length){
                H1(j,i)=h1(j)
            }
            
            //H2 of hidden    
            var h2=output(temp,W2,Bias)
            for(j<-0 until h2.length){
                H2(j,i)=h2(j)
            }
        }

         var H=kha_Rao_prod(H1,H2)
         var Y1=U1*H

         ////////////////////////////////////////////////////////////////////////
        //////////////////////////SECOND LAYER///////////////////////////////////
        /////////////////////////////////////////////////////////////////////// 

          var input2=DenseMatrix.vertcat(input,Y1)

        var H3=DenseMatrix.zeros[Double](W3.cols,input2.cols)
        var H4=DenseMatrix.zeros[Double](W4.cols,input2.cols)

        for(i<-0 until input2.cols){

            var temp=input2(::,i)
            //H3 of hidden  
            var h3=output(temp,W3,Bias)
            for(j<-0 until h3.length){
                H3(j,i)=h3(j)
            }
            
            //H4 of hidden    
            var h4=output(temp,W4,Bias)
            for(j<-0 until h4.length){
                H4(j,i)=h4(j)
            }
        }

         H=kha_Rao_prod(H3,H4)
         var Y2=U2*H

          ////////////////////////////////////////////////////////////////////////
        //////////////////////////THIRD LAYER///////////////////////////////////
        /////////////////////////////////////////////////////////////////////// 

       var input3=DenseMatrix.vertcat(input,Y2)
         var H5=DenseMatrix.zeros[Double](W5.cols,input3.cols)
        var H6=DenseMatrix.zeros[Double](W6.cols,input3.cols)

        for(i<-0 until input3.cols){

            var temp=input3(::,i)
            //H5 of hidden  
            var h5=output(temp,W5,Bias)
            for(j<-0 until h5.length){
                H5(j,i)=h5(j)
            }
            
            //H6 of hidden    
            var h6=output(temp,W6,Bias)
            for(j<-0 until h6.length){
                H6(j,i)=h6(j)
            }
        }

         H=kha_Rao_prod(H5,H6)
          var Y3=U3*H

          ////////////////////////////////////////////////////////////////////////
        //////////////////////////FOURTH LAYER///////////////////////////////////
        /////////////////////////////////////////////////////////////////////// 

        var input4=DenseMatrix.vertcat(input,Y3)

        var H7=DenseMatrix.zeros[Double](W7.cols,input4.cols)
        var H8=DenseMatrix.zeros[Double](W8.cols,input4.cols)

        for(i<-0 until input4.cols){

            var temp=input4(::,i)
            //H7 of hidden  
            var h7=output(temp,W7,Bias)
            for(j<-0 until h7.length){
                H7(j,i)=h7(j)
            }
            
            //H8 of hidden    
            var h8=output(temp,W8,Bias)
            for(j<-0 until h8.length){
                H8(j,i)=h8(j)
            }
        }

         H=kha_Rao_prod(H7,H8)
         var Y4=U4*H

         (Y4)

    }

