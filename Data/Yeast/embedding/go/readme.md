go的定义：
    go是用于编码蛋白质功能的词汇，一个蛋白质可以对应多个go编码
    所有的go编码组织成一张网络，用于描绘所有功能的从属关系和活动过程
go整体网络的下载：
    wget http://geneontology.org/ontology/go-basic.obo
    wget http://www.geneontology.org/ontology/subsets/goslim_generic.obo 注意goslim是对繁杂的go注释做归类之后的结果
    最后形成一个go-basic.obo的文件，里面描述了所有go的从属关系，是一个有向无环图
obo文件的解析：
    需要使用到goatools包，conda install goatools