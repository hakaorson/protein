计算蛋白质域相互作用，蛋白质域相互作用网络网站https://3did.irbbarcelona.org/ 下载3did_interface_flat.gz文件，为3did_interface_flat_Apr_10_2020.dat

    在dat文件中，每一个#=ID代表着一对相互作用域，其下有若干个#=IF，每一个#=IF代表一种拓扑形式

    通过文件可以构造蛋白质域相互作用网络，网络中一个结点代表一个拓朴域，一条边代表两个拓朴域存在相互作用，边的权重为拓扑的种类
    
    mapping：
        在uniprot下载的数据中只有拓朴域id，因此需要将拓朴域名称对应成拓朴域id
        这个对应关系可以从ftp://ftp.ebi.ac.uk/pub/databases/Pfam/mappings/ 下载
        其中PFAM_ACC即为拓朴域id（需要去除后面的小数点），PFAM_Name即为拓朴域名称
    
    最终可以得到如下的数据：
        拓朴域id1，拓朴域id2，作用强度
        所有的数据可以组织成一个graph，存储于domain_graph中
    

