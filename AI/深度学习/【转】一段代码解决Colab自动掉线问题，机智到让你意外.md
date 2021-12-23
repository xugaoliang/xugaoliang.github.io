# 【转】一段代码解决Colab自动掉线问题，机智到让你意外

程序员ShIvam Rawat在medium上贴出了一段代码：

```
function ClickConnect(){
console.log(“Working”);
document.querySelector(“colab-toolbar-button#connect”).click()
}
setInterval(ClickConnect,60000)
```
你只要把它放进控制台，它就会自动隔一阵儿调戏一下Colab页面，防止链接断掉。


---
**原文链接**
1. ShIvam Rawat : https://mp.weixin.qq.com/s/cdtfUyUpxtPE0xc32I4bPw
