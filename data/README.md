# Data Description

This directory contains data which record ILI activity levels in Italy, Germany, Belgium and the Netherlands starting from the 2006-2007 season to the 2018-2019 one.  
The data are taken from different sources. Usually, we tried to collect the data from national
health organization (e.g., Influnet and Influweb for Italy). If none of them was available, we resorted using the official WHO website.  
All these information are available online as convenient CSV files or PDF files that can be freely downloaded.
[Tabula](http://tabula.technology) software was used to extract data tables from the aforementioned PDF files.

The directory structure follows:

* **wikipedia_<country>**: each of those directory contains the Wikipedia pageviews and pagecounts data for that specific country. All of them contains the following data:
  * **new_data**: it contains the Wikipedia data ranging from 2007 to 2019. These data were made by merging together the pagecounts (2007-2015) and the pageviews (2015-2019).
  * **old_data**: it contains only the pagecounts Wikipedia data (2007-2015).
  * **pageviews**: it contains only the pageviews data (2015-2019).
  * **cyclerank**: it contains the Wikipedia data taken from the page selected by running the CycleRank algorithm. They range from 2007 to 2019 and they were made by combining the pageviews and pagerank data.
  * **cyclerank_pageviews**: it contains the Wikipedia data taken from the page selected by running the CycleRank algorithm. They range from 2015 to 2019 and they were made by using only the pageviews data.
  * **pagerank**: it contains the Wikipedia data taken from the page selected by running the PageRank algorithm. They range from 2007 to 2019 and they were made by combining the pageviews and pagerank data.
  * **pagerank_pageviews**: it contains the Wikipedia data taken from the page selected by running the PageRank algorithm. They range from 2015 to 2019 and they were made by using only the pageviews data.

* **<country>**: each of those directories contains the influenza incidence recorded for that particular country each year. The data collected ranges from 2007 to 2019. For some countries we have data starting from later years (e.g., 2010) and this is caused by the fact that they were not available anywhere. Inside each country directory, the data were divided into three directories: **new_data** (2007-2019), **old_data** (2007-2015) and **pageviews** (2015-2019).

* **keywords**: this directory uses the pages which were selected to be monitored. We have both the pages selected by the CycleRank and PageRank methods as well as the ones selected from the categories.

## Data format

### Wikipedia Pageviews and Pagecounts

Each CSV file contains as columns: the **year-week** and the list of all the selected pages. Moreover, if a certain cell is empty for a given year-week pair and page, it means that the page was not present in Wikipedia at that given time (maybe because it was added later).

### Influenza Incidence

Each CSV file contains two columns: **week** and **incidence**. Together, they give us information about the influenza rate during that given week. The files are also named this way: **<start_year>_<end_year>.csv** to give information about which influenza period they refer to. For instance, 2007_2008.csv stores data for the 2007-2008 influenza season.
