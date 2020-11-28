主要是将一系列的uniprotkb编码的蛋白质特性下载下来
注意可以参考的数据
基因序列 Sequence
    MIKNNCNNVN IYKYYLFSFK VYILPSNFKI WEAVSSMVSF KFLNLKPNNF LLFLLSRVAP RVLWVAPEPG FPTGILPFRT GKSSLRISRT SCIVNLASDT ISL
    需要将基因序列转换为保守型编码，参考https://www.ebi.ac.uk/Tools/sss/psiblast/
基因表达?
go注释编码 Gene ontology IDs
    GO:0004721; GO:0005739; GO:0005743; GO:0005744; GO:0016021; GO:0030150; GO:0030943; GO:0042802; GO:0046902
    其中有三种类型的编码
    1:Biological Process生物过程
    2:Molecular Function分子功能
    3:Cellular Component细胞组成
subcell定位 Subcellular location [CC]
    SUBCELLULAR LOCATION: Nucleus, nucleolus {ECO:0000269|PubMed:9632712}
蛋白质域相互作用
    有两种域数据集，分别是
    Cross-reference (Pfam)
        示例：PF01798;PF08156;
    Cross-reference (InterPro)
        示例：IPR029012;IPR012974;IPR042239;IPR002687;IPR036070;IPR012976;

所有的编码数据都是根据特定的蛋白质网络计算过来的

技巧
    在uniprot中可以下载到所有的蛋白质，操作是在advanced搜索里面做空白搜索即可（一共56万个）
    在download的时候下载map，可以只下载id map数据

注意计算特征的时候是无关有向图和无向图的
