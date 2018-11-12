## *比较岭回归和Lasso回归的区别*
  
岭回归与Lasso回归的出现是为了解决线性回归出现的过拟合以及在通过正规方程方法求解θ的过程中出现的x转置乘以x不可逆这两类问题的，这两种回归均通过在损失函数中引入正则化项来达到目的，具体三者的损失函数对比如下： 
线性回归的损失公式：<p align="center"><img src="https://latex.codecogs.com/gif.latex?J(&#x5C;theta)=&#x5C;frac{1}{2m}&#x5C;sum&#x5C;limits_{i=1}^n[h_&#x5C;theta(%20x^(i)%20)-y^(i)]^2"/></p>  
  
  
岭回归的损失函数：<p align="center"><img src="https://latex.codecogs.com/gif.latex?J（&#x5C;theta）=&#x5C;frac{1}{2m}&#x5C;sum&#x5C;limits_{i=1}^n[h_&#x5C;theta(%20x^(i)%20)-y^(i)]^2+&#x5C;lambda&#x5C;sum&#x5C;limits_{j=1}^n&#x5C;theta_j^2"/></p>  
  
  
Lasso回归的损失函数：<p align="center"><img src="https://latex.codecogs.com/gif.latex?J(&#x5C;theta)=&#x5C;frac{1}{2m}&#x5C;sum&#x5C;limits_{i=1}^n[h_&#x5C;theta(%20x^(i)%20)-y^(i)]^2+&#x5C;lambda&#x5C;sum&#x5C;limits_{j=1}^n|&#x5C;theta_j|"/></p>  
  
  
其中λ称为正则化参数，如果λ选取过大，会把所有参数θ均最小化，造成欠拟合，如果λ选取过小，会导致对过拟合问题解决不当，因此λ的选取是一个技术活。 岭回归与Lasso回归最大的区别在于岭回归引入的是L2范数惩罚项，Lasso回归引入的是L1范数惩罚项，Lasso回归能够使得损失函数中的许多θ均变成0，这点要优于岭回归，因为岭回归是要所有的θ均存在的，这样计算量Lasso回归将远远小于岭回归。 
  
Ridge:![Ridge](https://img-blog.csdn.net/20170815211707575?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvaHp3MTk5MjAzMjk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast )
  
Lasso:![Lasso](https://img-blog.csdn.net/20170815212210169?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvaHp3MTk5MjAzMjk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast )
  
可以看到，Lasso回归最终会趋于一条直线，原因就在于好多θ值已经均为0，而岭回归却有一定平滑度，因为所有的θ值均存在。
  