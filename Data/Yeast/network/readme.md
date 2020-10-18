DIP官网数据：https://dip.doe-mbi.ucla.edu/dip/Main.cgi
将dip数据集转换为uniprotkb编码，有一些条目不具有uniprotkb编码

    不具备uniprotkb编码的条目，不存在uniprotkb这个条目：
    DIP-2517N|refseq:NP_011700|uniprotkb:P19812	DIP-59062N
    DIP-2517N|refseq:NP_011700|uniprotkb:P19812	DIP-59065N

    uniprotkb编码的格式：
    P19812

舍弃不具备uniprotkb编码的条目69条，自循环条目353条,重复条目3条，最后剩余22552条数据（原来22977条），存放于dip_22977中

有一些id在uniprotkb条目存在，但是在uniprotkb网站上无法匹配(非Baker's yeast)：
所有的id存放在dip_ids中，在uniprotkb匹配之后能够对应的条目存放在dip_id_mapped中
最终需要从dip_22977中去除一部分无法匹配的数据560条
最终剩余21992条数据，存放于dip中，最终独立的蛋白质id数目为4933个

