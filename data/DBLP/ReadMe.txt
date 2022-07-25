This directory contains the DBLP citation network, a subgraph extracted from DBLP-Citation-network V3 (https://aminer.org/citation).

To form this subgraph, papers from four subjects are extracted according to their venue information: Database, Data Mining, Artificial Intelligence and Computer Vision, and papers with no citations are removed.

The DBLP network contains 18,448 papers and 45,661 citation relations. From paper titles, we construct 2,476-dimensional binary node feature vectors, with each element indicating the presence/absence of the corresponding word.

By ignoring the citation direction, we take the DBLP subgraph as an undirected network.

THE DIRECTORY CONTAINS TWO FILES:

The content.txt file contains descriptions of the papers in the following format:

		<paper_id> <word_attributes> <class_label> <publication_year>

The first entry in each line contains the unique integer ID (ranging from 0 to 18,447) of the paper followed by binary values indicating whether each word in the vocabulary is present (indicated by 1) or absent (indicated by 0) in the paper. Finally, the last two entries in the line are the class label and the publication year of the paper.

The edgeList.txt file contains the citation relations. Each line describes a link in the following format:

		<ID of paper1> <ID of paper2>

Each line contains two paper IDs, with paper2 citing paper1 or paper1 citing paper2.
