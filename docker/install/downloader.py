from sklearn.datasets import fetch_openml
from tqdm import tqdm
from sklearn import datasets
import torchvision
from transformers import AutoTokenizer, AutoModelForMaskedLM

openml_datasets = [
    "123123\tone-hundred-plants-margin	1	ARFF	1600	65	100	0	143808	1	17	18	418	active	2015-05-25 20:51:03	https://www.openml.org/data/download/1592283/phpCsX3fx",
    "1457	amazon-commerce-reviews	1	ARFF	1500	10001	50	0	65168	2	49	51	216	active	2015-05-21 16:25:44	https://www.openml.org/data/download/1586202/phpr1uf8O",
    "300	isolet	1	ARFF	7797	618	26	0	50879	0	71	71	132	active	2014-08-20 20:59:05	https://www.openml.org/data/download/52405/phpB0xrNj",
    "1515	micro-mass	2	ARFF	571	1301	20	0	39938	1	17	18	98	active	2015-06-01 16:57:25	https://www.openml.org/data/download/1593707/phpHyLSNF",
    "333	monks-problems-1	1	ARFF	556	7	2	0	358750	2	22	24	41	active	2014-08-26 17:11:18	https://www.openml.org/data/download/52236/phpAyyBys",
    "334	monks-problems-2	1	ARFF	601	7	2	0	394564	3	34	37	38	active	2014-08-26 17:29:02	https://www.openml.org/data/download/52237/php4fATLZ",
    "335	monks-problems-3	1	ARFF	554	7	2	0	108816	1	15	16	35	active	2014-08-26 17:41:07	https://www.openml.org/data/download/52238/phphZierv",
    "4134	Bioresponse	1	ARFF	3751	1777	2	0	48676	2	40	42	34	active	2015-11-04 23:59:56	https://www.openml.org/data/download/1681097/phpSSK7iA",
    "1116	musk	1	ARFF	6598	168	2	0	82516	1	20	21	33	active	2014-10-07 00:41:54	https://www.openml.org/data/download/53999/musk.arff",
    "31	credit-g	1	ARFF	1000	21	2	0	506283	28	311	339	32	active	2014-04-06 23:21:47	https://www.openml.org/data/download/31/dataset_31_credit-g.arff",
    "1462	banknote-authentication	1	ARFF	1372	5	2	0	138167	6	40	46	32	active	2015-05-21 22:40:57	https://www.openml.org/data/download/1586223/php50jXam",
    "1471	eeg-eye-state	1	ARFF	14980	15	2	0	166145	3	96	99	29	active	2015-05-22 16:40:04	https://www.openml.org/data/download/1587924/phplE7q6h",
    "1067	kc1	1	ARFF	2109	22	2	0	161516	2	29	31	29	active	2014-10-06 23:57:43	https://www.openml.org/data/download/53950/kc1.arff",
    "1494	qsar-biodeg	1	ARFF	1055	42	2	0	267858	1	25	26	28	active	2015-05-25 21:14:53	https://www.openml.org/data/download/1592286/phpGUrE90",
    "1487	ozone-level-8hr	1	ARFF	2534	73	2	0	188259	1	20	21	28	active	2015-05-25 19:22:29	https://www.openml.org/data/download/1592279/phpdReP6S",
    "1046	mozilla4	1	ARFF	15545	6	2	0	109963	1	21	22	28	active	2014-10-06 23:57:07	https://www.openml.org/data/download/53929/mozilla4.arff",
    "1510	wdbc	1	ARFF	569	31	2	0	226889	4	39	43	27	active	2015-05-26 16:24:07	https://www.openml.org/data/download/1592318/phpAmSP4g",
    "1068	pc1	1	ARFF	1109	22	2	0	149998	0	27	27	27	active	2014-10-06 23:57:45	https://www.openml.org/data/download/53951/pc1.arff",
    "1049	pc4	1	ARFF	1458	38	2	0	115699	0	17	17	27	active	2014-10-06 23:57:12	https://www.openml.org/data/download/53932/pc4.arff",
    "4534	PhishingWebsites	1	ARFF	11055	31	2	0	51670	1	29	30	27	active	2016-02-16 15:30:33	https://www.openml.org/data/download/1798106/phpV5QYya",
    "1063	kc2	1	ARFF	522	22	2	0	176897	0	27	27	26	active	2014-10-06 23:57:36	https://www.openml.org/data/download/53946/kc2.arff",
    "1480	ilpd	1	ARFF	583	11	2	0	155160	2	24	26	26	active	2015-05-22 22:40:56	https://www.openml.org/data/download/1590565/phpOJxGL9",
    "1050	pc3	1	ARFF	1563	38	2	0	146025	1	18	19	26	active	2014-10-06 23:57:13	https://www.openml.org/data/download/53933/pc3.arff",
    "1485	madelon	1	ARFF	2600	501	2	0	101200	0	20	20	26	active	2015-05-22 23:46:18	https://www.openml.org/data/download/1590986/phpfLuQE4",
    "1038	gina_agnostic	1	ARFF	3468	971	2	0	69231	0	21	21	26	active	2014-10-06 23:56:01	https://www.openml.org/data/download/53921/gina_agnostic.arff",
    "1120	MagicTelescope	1	ARFF	19020	12	2	0	64979	1	33	34	26	active	2014-10-07 00:42:01	https://www.openml.org/data/download/54003/MagicTelescope.arff",
    "1504	steel-plates-fault	1	ARFF	1941	34	2	0	277764	2	52	54	25	active	2015-05-25 22:42:40	https://www.openml.org/data/download/1592296/php9xWOpn",
    "1479	hill-valley	1	ARFF	1212	101	2	0	183564	0	24	24	25	active	2015-05-22 21:11:58	https://www.openml.org/data/download/1590101/php3isjYz",
    "312	scene	1	ARFF	2407	300	2	0	90238	0	24	24	25	active	2014-08-25 11:43:22	https://www.openml.org/data/download/1390080/phpuZu33P",
    "37	diabetes	1	ARFF	768	9	2	0	202456	8	107	115	19	active	2014-04-06 23:22:13	https://www.openml.org/data/download/37/dataset_37_diabetes.arff",
    "3	kr-vs-kp	1	ARFF	3196	37	2	0	274202	1	44	45	16	active	2014-04-06 23:19:28	https://www.openml.org/data/download/3/dataset_3_kr-vs-kp.arff",
    "12	mfeat-factors	1	ARFF	2000	217	10	0	37684	0	18	18	15	active	2014-04-06 23:20:04	https://www.openml.org/data/download/12/dataset_12_mfeat-factors.arff",
    "42	soybean	1	ARFF	683	36	19	2337	41045	1	56	57	13	active	2014-04-06 23:22:32	https://www.openml.org/data/download/42/dataset_42_soybean.arff",
    "14	mfeat-fourier	1	ARFF	2000	77	10	0	38337	0	13	13	13	active	2014-04-06 23:20:17	https://www.openml.org/data/download/14/dataset_14_mfeat-fourier.arff",
    "44	spambase	1	ARFF	4601	58	2	0	161987	5	93	98	12	active	2014-04-06 23:22:41	https://www.openml.org/data/download/44/dataset_44_spambase.arff",
    "151	electricity	1	ARFF	45312	9	2	0	107167	3	45	48	12	active	2014-04-10 02:42:23	https://www.openml.org/data/download/2419/electricity-normalized.arff",
    "6	letter	1	ARFF	20000	17	26	0	69575	3	75	78	12	active	2014-04-06 23:19:41	https://www.openml.org/data/download/6/dataset_6_letter.arff",
    "16	mfeat-karhunen	1	ARFF	2000	65	10	0	38799	0	20	20	12	active	2014-04-06 23:20:30	https://www.openml.org/data/download/16/dataset_16_mfeat-karhunen.arff",
    "32	pendigits	1	ARFF	10992	17	10	0	37564	0	21	21	12	active	2014-04-06 23:21:54	https://www.openml.org/data/download/32/dataset_32_pendigits.arff",
    "28	optdigits	1	ARFF	5620	65	10	0	36107	3	22	25	12	active	2014-04-06 23:21:34	https://www.openml.org/data/download/28/dataset_28_optdigits.arff",
    "18	mfeat-morphological	1	ARFF	2000	7	10	0	35656	1	19	20	12	active	2014-04-06 23:20:37	https://www.openml.org/data/download/18/dataset_18_mfeat-morphological.arff",
    "50	tic-tac-toe"
]

openml_datasets = [d.split('\t')[1] for d in openml_datasets]

downloaded = {}

# for d in tqdm(openml_datasets):
#     print(d)
#     downloaded[d] = fetch_openml(d, cache=True)


downloaded['20newsgroups'] = datasets.fetch_20newsgroups()
downloaded['20newsgroups_vectorized'] = datasets.fetch_20newsgroups_vectorized()
downloaded['california_housing'] = datasets.fetch_california_housing()
downloaded['covtype'] = datasets.fetch_covtype()
downloaded['kddcup99'] = datasets.fetch_kddcup99(percent10=False, as_frame=True)
# downloaded['lfw_pairs'] = datasets.fetch_lfw_pairs()
# downloaded['lfw_people'] = datasets.fetch_lfw_people()
# downloaded['olivetti_faces'] = datasets.fetch_olivetti_faces()
# downloaded['rcv1'] = datasets.fetch_rcv1()
# downloaded['species_distributions'] = datasets.fetch_species_distributions()


pytorch_data = '/root/pytorch_data'

pytorch_datasets = ['CIFAR10', 'CIFAR100', 'FashionMNIST', 'MNIST']

for d in pytorch_datasets:
    print(d)
    f = getattr(torchvision.datasets, d)
    f(pytorch_data, train=True, download=True)
    f(pytorch_data, train=False, download=True)


model_names = ["bert-base-cased", "aubmindlab/bert-base-arabertv02", "xlm-roberta-base",
               "bert-base-multilingual-cased", "asafaya/bert-mini-arabic"]

for m in model_names:
    tokenizer = AutoTokenizer.from_pretrained(m)
    print(tokenizer)
    try:
        model = AutoModelForMaskedLM.from_pretrained(m)
        print(model)
    except Exception as e:
        print(e)


