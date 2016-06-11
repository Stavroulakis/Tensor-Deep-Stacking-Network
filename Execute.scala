////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
/////////////////TRAINING DATA ////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
var a=sc.textFile("hdfs:/tmp/alex/set12_train1.csv")
var s=a.collect()
//var d :Array[String]= new Array[String](190)
var test=s(1).split(",")
var X=DenseMatrix.zeros[Double](test.length ,s.length)

for(i<-0 until s.length){
     test=s(i).split(",")
    for(j<-0 until test.length){
        X(j,i)=test(j).toDouble
    }

}

///target
var T=DenseMatrix.zeros[Double](1,X.cols)
for(i<-0 until T.cols){
    if(i < 77){
        T(0,i)=(-1.0)
    }
    else if (i < 90) {
          T(0,i)=0.0
    }
    else{
         T(0,i)=1.0
    }
}


//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
//////////////////IMPORTANT MATRICES/////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
var Xn=DenseMatrix.vertcat(X,T)
var wlength=25
var w11=DenseMatrix.zeros[Double](X.rows,wlength)
var w22=DenseMatrix.zeros[Double](X.rows,wlength)
var w33=DenseMatrix.zeros[Double](Xn.rows,wlength)
var w44=DenseMatrix.zeros[Double](Xn.rows,wlength)
var w55=DenseMatrix.zeros[Double](Xn.rows,wlength)
var w66=DenseMatrix.zeros[Double](Xn.rows,wlength)
var w77=DenseMatrix.zeros[Double](Xn.rows,wlength)
var w88=DenseMatrix.zeros[Double](Xn.rows,wlength)

////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////TRAIN/////////////////////////////////////////////
var (w1,w2,u1,y1,w3,w4,u2,y2,w5,w6,u3,y3,w7,w8,u4,y4,er1,er2,er3,er4)=training_network(X,T,2.0,1.0,7.0,wlength,w11,w22,w33,w44,w55,w66,w77,w88,0)


////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
//////////////////////DATA FOR PREDICTION//////////////////////////////////
//////////////////////////////////////////////////////////////////////////
var a=sc.textFile("hdfs:/tmp/alex/testSets/testset11.csv")
var s=a.collect()
var test=s(1).split(",")
var Xtest=DenseMatrix.zeros[Double](test.length ,s.length)

for(i<-0 until s.length){
     test=s(i).split(",")
    for(j<-0 until test.length ){
        Xtest(j,i)=test(j).toDouble
    }
}

////////////////////////////////////////////////////////////

var a=sc.textFile("hdfs:/tmp/alex/testSets/testset5.csv")
var s=a.collect()
var test=s(1).split(",")
var Xtest=DenseMatrix.zeros[Double](test.length ,s.length)

for(i<-0 until s.length){
     test=s(i).split(",")
    for(j<-0 until test.length ){
        Xtest(j,i)=test(j).toDouble
    }
}

///////////////////////////////////////////////////////////////

var a=sc.textFile("hdfs:/tmp/alex/testSets/testset2.csv")
var s=a.collect()
var test=s(1).split(",")
var Xtest=DenseMatrix.zeros[Double](test.length ,s.length)

for(i<-0 until s.length){
     test=s(i).split(",")
    for(j<-0 until test.length ){
        Xtest(j,i)=test(j).toDouble
    }
}

/////////////////////////////////////////////////////////////
var a=sc.textFile("hdfs:/tmp/alex/testSets/testset3.csv")
var s=a.collect()
var test=s(1).split(",")
var Xtest=DenseMatrix.zeros[Double](test.length ,s.length)

for(i<-0 until s.length){
     test=s(i).split(",")
    for(j<-0 until test.length ){
        Xtest(j,i)=test(j).toDouble
    }
}

////////////////////////////////////////////////////////////

var a=sc.textFile("hdfs:/tmp/alex/testSets/testset6.csv")
var s=a.collect()
var test=s(1).split(",")
var Xtest=DenseMatrix.zeros[Double](test.length ,s.length)

for(i<-0 until s.length){
     test=s(i).split(",")
    for(j<-0 until test.length ){
        Xtest(j,i)=test(j).toDouble
    }
}


/////////////////////////////////////////////////////////////

var a=sc.textFile("hdfs:/tmp/alex/testSets/testset7.csv")
var s=a.collect()
var test=s(1).split(",")
var Xtest=DenseMatrix.zeros[Double](test.length ,s.length)

for(i<-0 until s.length){
     test=s(i).split(",")
    for(j<-0 until test.length ){
        Xtest(j,i)=test(j).toDouble
    }
}


/////////////////////////////////////////////////////////////

var a=sc.textFile("hdfs:/tmp/alex/testSets/testset8.csv")
var s=a.collect()
var test=s(1).split(",")
var Xtest=DenseMatrix.zeros[Double](test.length ,s.length)

for(i<-0 until s.length){
     test=s(i).split(",")
    for(j<-0 until test.length ){
        Xtest(j,i)=test(j).toDouble
    }
}

//////////////////////////////////////////////////////////

var a=sc.textFile("hdfs:/tmp/alex/testSets/testset9.csv")
var s=a.collect()
var test=s(1).split(",")
var Xtest=DenseMatrix.zeros[Double](test.length ,s.length)

for(i<-0 until s.length){
     test=s(i).split(",")
    for(j<-0 until test.length ){
        Xtest(j,i)=test(j).toDouble
    }
}

///////////////////////////////////////////////////////////

var a=sc.textFile("hdfs:/tmp/alex/testSets/testset10.csv")
var s=a.collect()
var test=s(1).split(",")
var Xtest=DenseMatrix.zeros[Double](test.length ,s.length)

for(i<-0 until s.length){
     test=s(i).split(",")
    for(j<-0 until test.length ){
        Xtest(j,i)=test(j).toDouble
    }
}

//////////////////////////////////////////////////////////

var a=sc.textFile("hdfs:/tmp/alex/testSets/testset12.csv")
var s=a.collect()
var test=s(1).split(",")
var Xtest=DenseMatrix.zeros[Double](test.length ,s.length)

for(i<-0 until s.length){
     test=s(i).split(",")
    for(j<-0 until test.length ){
        Xtest(j,i)=test(j).toDouble
    }
}


////////////////////////////////////////////////////////

var a=sc.textFile("hdfs:/tmp/alex/testSets/bigtest.csv")
var s=a.collect()
var test=s(1).split(",")
var Xtest=DenseMatrix.zeros[Double](test.length ,s.length)

for(i<-0 until s.length){
     test=s(i).split(",")
    for(j<-0 until test.length ){
        Xtest(j,i)=test(j).toDouble
    }
}



/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////PREDICT/////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

var telos=predict(Xtest,w1,w2,u1,w3,w4,u2,w5,w6,u3,w7,w8,u4,1.0)

///REGULATE RESULTS//////////////////////////////////////////////
for(i<-0 until telos.cols){
    if(telos(0,i)>0.5){
        telos(0,i)=1.0
    }
    else if(telos(0,i)<(-0.5)){
        telos(0,i)=(-1.0)
    }
    else{
        telos(0,i)=0
    }
}