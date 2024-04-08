from example_utils import add_beam_to_path
add_beam_to_path()

from src.beam.llm import beam_llm

# llm = beam_llm('openai:///gpt-4')

# res = llm.chat("hi how are you?", system="you had a bad day")
# print(res)

# gen = llm.ask('how are you?', stream=True)
#
# r = next(gen)

from src.beam.llm.simulators.openai import OpenAI

client = OpenAI()

completion = client.chat.completions.create(
  model='openai:///gpt-4',
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!, count to 3:"}
  ],
  stream=True
)

for chunk in completion:
  print(chunk)

# from src.beam.llm import text_splitter
#
#
# text = """
# The cat (Felis catus) is a domestic species of small carnivorous mammal.[1][2] It is the only domesticated species in the family Felidae and is commonly referred to as the domestic cat or house cat to distinguish it from the wild members of the family.[4] Cats are commonly kept as house pets but can also be farm cats or feral cats; the feral cat ranges freely and avoids human contact.[5] Domestic cats are valued by humans for companionship and their ability to kill vermin. About 60 cat breeds are recognized by various cat registries.[6]
#
# The cat is similar in anatomy to the other felid species: it has a strong flexible body, quick reflexes, sharp teeth, and retractable claws adapted to killing small prey like mice and rats. Its night vision and sense of smell are well developed. Cat communication includes vocalizations like meowing, purring, trilling, hissing, growling, and grunting as well as cat-specific body language. Although the cat is a social species, it is a solitary hunter. As a predator, it is crepuscular, i.e. most active at dawn and dusk. It can hear sounds too faint or too high in frequency for human ears, such as those made by mice and other small mammals.[7] It also secretes and perceives pheromones.[8]
#
# Female domestic cats can have kittens from spring to late autumn in temperate zones and throughout the year in equatorial regions, with litter sizes often ranging from two to five kittens.[9][10] Domestic cats are bred and shown at events as registered pedigreed cats, a hobby known as cat fancy. Population control of cats may be achieved by spaying and neutering, but their proliferation and the abandonment of pets has resulted in large numbers of feral cats worldwide, contributing to the extinction of entire bird, mammal, and reptile species.[11]
#
# It was long thought that cat domestication began in ancient Egypt, where cats were venerated from around 3100 BC,[12][13] but recent advances in archaeology and genetics have shown that their domestication occurred in the Near East around 7500 BC.[14]
#
# As of 2021, there were an estimated 220 million owned and 480 million stray cats in the world.[15][16] As of 2017, the domestic cat was the second most popular pet in the United States, with 95.6 million cats owned[17][18] and around 42 million households owning at least one cat.[19] In the United Kingdom, 26% of adults have a cat, with an estimated population of 10.9 million pet cats as of 2020.[20]
#
# Etymology and naming
# The origin of the English word cat, Old English catt, is thought to be the Late Latin word cattus, which was first used at the beginning of the 6th century.[21] The Late Latin word may be derived from an unidentified African language.[22] The Nubian word kaddîska 'wildcat' and Nobiin kadīs are possible sources or cognates.[23] The Nubian word may be a loan from Arabic قَطّ‎ qaṭṭ ~ قِطّ qiṭṭ.[citation needed]
#
# However, it is "equally likely that the forms might derive from an ancient Germanic word, imported into Latin and thence to Greek and to Syriac and Arabic".[24] The word may be derived from Germanic and Northern European languages, and ultimately be borrowed from Uralic, cf. Northern Sami gáđfi, 'female stoat', and Hungarian hölgy, 'lady, female stoat'; from Proto-Uralic *käďwä, 'female (of a furred animal)'.[25]
#
# The English puss, extended as pussy and pussycat, is attested from the 16th century and may have been introduced from Dutch poes or from Low German puuskatte, related to Swedish kattepus, or Norwegian pus, pusekatt. Similar forms exist in Lithuanian puižė and Irish puisín or puiscín. The etymology of this word is unknown, but it may have arisen from a sound used to attract a cat.[26][27]
#
# A male cat is called a tom or tomcat[28] (or a gib,[29] if neutered). A female is called a queen[30] (or a molly,[31][user-generated source?] if spayed), especially in a cat-breeding context. A juvenile cat is referred to as a kitten. In Early Modern English, the word kitten was interchangeable with the now-obsolete word catling.[32] A group of cats can be referred to as a clowder or a glaring.[33]
#
# Taxonomy
# The scientific name Felis catus was proposed by Carl Linnaeus in 1758 for a domestic cat.[1][2] Felis catus domesticus was proposed by Johann Christian Polycarp Erxleben in 1777.[3] Felis daemon proposed by Konstantin Satunin in 1904 was a black cat from the Transcaucasus, later identified as a domestic cat.[34][35]
#
# In 2003, the International Commission on Zoological Nomenclature ruled that the domestic cat is a distinct species, namely Felis catus.[36][37] In 2007, it was considered a subspecies, F. silvestris catus, of the European wildcat (F. silvestris) following results of phylogenetic research.[38][39] In 2017, the IUCN Cat Classification Taskforce followed the recommendation of the ICZN in regarding the domestic cat as a distinct species, Felis catus.[40]
#
# Evolution
# Main article: Cat evolution
#
# Skulls of a wildcat (top left), a housecat (top right), and a hybrid between the two. (bottom center)
# The domestic cat is a member of the Felidae, a family that had a common ancestor about 10–15 million years ago.[41] The genus Felis diverged from other Felidae around 6–7 million years ago.[42] Results of phylogenetic research confirm that the wild Felis species evolved through sympatric or parapatric speciation, whereas the domestic cat evolved through artificial selection.[43] The domesticated cat and its closest wild ancestor are diploid and both possess 38 chromosomes[44] and roughly 20,000 genes.[45] The leopard cat (Prionailurus bengalensis) was tamed independently in China around 5500 BC. This line of partially domesticated cats leaves no trace in the domestic cat populations of today.[46]
#
# Domestication
# See also: Domestication of the cat
#
# A cat eating a fish under a chair, a mural in an Egyptian tomb dating to the 15th century BC
# The earliest known indication for the taming of an African wildcat (F. lybica) was excavated close by a human Neolithic grave in Shillourokambos, southern Cyprus, dating to about 7500–7200 BC. Since there is no evidence of native mammalian fauna on Cyprus, the inhabitants of this Neolithic village most likely brought the cat and other wild mammals to the island from the Middle Eastern mainland.[47] Scientists therefore assume that African wildcats were attracted to early human settlements in the Fertile Crescent by rodents, in particular the house mouse (Mus musculus), and were tamed by Neolithic farmers. This mutual relationship between early farmers and tamed cats lasted thousands of years. As agricultural practices spread, so did tame and domesticated cats.[14][6] Wildcats of Egypt contributed to the maternal gene pool of the domestic cat at a later time.[48]
#
# The earliest known evidence for the occurrence of the domestic cat in Greece dates to around 1200 BC. Greek, Phoenician, Carthaginian and Etruscan traders introduced domestic cats to southern Europe.[49] During the Roman Empire they were introduced to Corsica and Sardinia before the beginning of the 1st millennium.[50] By the 5th century BC, they were familiar animals around settlements in Magna Graecia and Etruria.[51] By the end of the Western Roman Empire in the 5th century, the Egyptian domestic cat lineage had arrived in a Baltic Sea port in northern Germany.[48]
#
# During domestication, cats have undergone only minor changes in anatomy and behavior, and they are still capable of surviving in the wild. Several natural behaviors and characteristics of wildcats may have pre-adapted them for domestication as pets. These traits include their small size, social nature, obvious body language, love of play, and high intelligence. Captive Leopardus cats may also display affectionate behavior toward humans but were not domesticated.[52] House cats often mate with feral cats.[53] Hybridisation between domestic and other Felinae species is also possible, producing hybrids such as the Kellas cat in Scotland.[54][55]
#
# Development of cat breeds started in the mid 19th century.[56] An analysis of the domestic cat genome revealed that the ancestral wildcat genome was significantly altered in the process of domestication, as specific mutations were selected to develop cat breeds.[57] Most breeds are founded on random-bred domestic cats. Genetic diversity of these breeds varies between regions, and is lowest in purebred populations, which show more than 20 deleterious genetic disorders.[58]
#
# Characteristics
# Main article: Cat anatomy
# Size
#
# Diagram of the general anatomy of a male domestic cat
# The domestic cat has a smaller skull and shorter bones than the European wildcat.[59] It averages about 46 cm (18 in) in head-to-body length and 23–25 cm (9–10 in) in height, with about 30 cm (12 in) long tails. Males are larger than females.[60] Adult domestic cats typically weigh between 4 and 5 kg (9 and 11 lb).[43]
#
# Skeleton
# Cats have seven cervical vertebrae (as do most mammals); 13 thoracic vertebrae (humans have 12); seven lumbar vertebrae (humans have five); three sacral vertebrae (as do most mammals, but humans have five); and a variable number of caudal vertebrae in the tail (humans have only three to five vestigial caudal vertebrae, fused into an internal coccyx).[61]: 11  The extra lumbar and thoracic vertebrae account for the cat's spinal mobility and flexibility. Attached to the spine are 13 ribs, the shoulder, and the pelvis.[61]: 16  Unlike human arms, cat forelimbs are attached to the shoulder by free-floating clavicle bones which allow them to pass their body through any space into which they can fit their head.[62]
# """
#
#
# if __name__ == '__main__':
#
#     # llm = beam_llm("tgi://192.168.10.45:40081")
#     #
#     # small_chat_example = ['hi my name is elad', 'what your name?', 'how is the weather in london?',
#     #                       'do you remember my name?']
#     #
#     # for t in small_chat_example:
#     #     print(f"User: {t}")
#     #     print(f"LLM: {llm.chat(t).text}")
#
#
#     # print(llm.chat('hi my name is elad').text)
#     # llm = beam_llm("tgi://192.168.10.45:40081")
#     # print(llm.chat('do you remember my name?').text)
#
#     chunks = text_splitter(text, chunk_size=200)
#
#     for i, c in enumerate(chunks):
#         print(f"Chunk {i}:")
#         print(c)
#         print()



