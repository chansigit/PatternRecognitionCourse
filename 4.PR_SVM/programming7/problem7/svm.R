# Read data
setwd('E:/博一下/课程-模式识别/第四次作业 SVM/Programming/problem7')
x <- as.data.frame(read.table("data.txt", header=T))
x$category <- as.factor(x$category)

# Preprocess data: appending extra dimensions
x$'1' <- 1
x$'x1^2' <- (x$x1)^2
x$'x2^2' <- (x$x2)^2
x$'x1x2' <- (x$x1)*(x$x2)
x <- x[, c('category','1','x1','x2','x1x2','x1^2','x2^2')]

# Extract trainning set
w1 <- x[x$category==1,]
w2 <- x[x$category==2,]

trainSet1 <- rbind(w1[1:1, ],w2[1:1, ])
trainSet2 <- rbind(w1[1:2, ],w2[1:2, ])
trainSet3 <- rbind(w1[1:3, ],w2[1:3, ])
trainSet4 <- rbind(w1[1:4, ],w2[1:4, ])


#plot(x$'x1' , x$'x2', col=x$'category', type = 'p', cex=2, pch=16,
#     xlab="X1",ylab="X2")


# Train and Validate
library(e1071)

#----------------------------------------------------------
## train with first 1 points
svm_model <- svm(trainSet1$category ~ ., trainSet1,kernel="linear")
summary(svm_model)
# show weights of hyperplane
t(svm_model$coefs)%*% svm_model$SV
# show negative intercept of hyperplane
svm_model$rho

predicted <- predict(svm_model, x)
validated <- (predicted==x$category)
sprintf("misclassified=%d/%d, ratio=%f", 
        length(validated[validated==FALSE]),
        length(predicted),
        length(validated[validated==FALSE])/length(predicted))


#----------------------------------------------------------
## train with first 2 points
svm_model <- svm(trainSet2$category ~ ., trainSet2,kernel="linear")
summary(svm_model)
# show weights of hyperplane
t(svm_model$coefs)%*% svm_model$SV
# show negative intercept of hyperplane
svm_model$rho

predicted <- predict(svm_model, x)
validated <- (predicted==x$category)
sprintf("misclassified=%d/%d, ratio=%f", 
        length(validated[validated==FALSE]),
        length(predicted),
        length(validated[validated==FALSE])/length(predicted))


#----------------------------------------------------------
## train with first 3 points
svm_model <- svm(trainSet3$category ~ ., trainSet3,kernel="linear")
summary(svm_model)
# show weights of hyperplane
t(svm_model$coefs)%*% svm_model$SV
# show negative intercept of hyperplane
svm_model$rho

predicted <- predict(svm_model, x)
validated <- (predicted==x$category)
sprintf("misclassified=%d/%d, ratio=%f", 
        length(validated[validated==FALSE]),
        length(predicted),
        length(validated[validated==FALSE])/length(predicted))


#----------------------------------------------------------
## train with first 4 points
svm_model <- svm(trainSet4$category ~ ., trainSet4,kernel="linear")
summary(svm_model)
# show weights of hyperplane
t(svm_model$coefs)%*% svm_model$SV
# show negative intercept of hyperplane
svm_model$rho

predicted <- predict(svm_model, x)
validated <- (predicted==x$category)
sprintf("misclassified=%d/%d, ratio=%f", 
        length(validated[validated==FALSE]),
        length(predicted),
        length(validated[validated==FALSE])/length(predicted))

#----------------------------------------------------------
## train with all points
svm_model <- svm(x$category ~ ., x,kernel="linear")
summary(svm_model)
# show weights of hyperplane
t(svm_model$coefs)%*% svm_model$SV
# show negative intercept of hyperplane
svm_model$rho

predicted <- predict(svm_model, x)
validated <- (predicted==x$category)
sprintf("misclassified=%d/%d, ratio=%f", 
        length(validated[validated==FALSE]),
        length(predicted),
        length(validated[validated==FALSE])/length(predicted))

#----------------------------------------------------------
library(rgl)
plot3d(x$'x1', x$'x2', x$'x1x2', col=x$'category',
       xlab='x1', ylab='x2',zlab='x1x2', type='s', size=2.4)

plot3d(x$'x1', x$'x2', x$'x1^2', col=x$'category',
       xlab='x1', ylab='x2',zlab='x1^2', type='s', size=2.4)

plot3d(x$'x1', x$'x2', x$'x2^2', col=x$'category',
       xlab='x1', ylab='x2',zlab='x2^2', type='s', size=2.4)

plot3d(x$'x1', x$'x1^2', x$'x2^2', col=x$'category',
       xlab='x1', ylab='x1^2',zlab='x2^2', type='s', size=2.4)

plot3d(x$'x1^2', x$'x2^2', x$'x1x2', col=x$'category',
       xlab='x1^2', ylab='x2^2',zlab='x1x2', type='s', size=2.4)