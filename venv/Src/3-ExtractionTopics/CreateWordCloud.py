import sys, os, math
sys.path.append('..')
from DBConnection.MongoDBConnection import MongoDB
from wordcloud import WordCloud
from fpdf import FPDF
import matplotlib.pyplot as plt

with MongoDB("TesiMagistrale") as db:
    pdf = FPDF()
    pdf.set_font('Times', size=14)

    for temporalSlice in db.query("TemporalSlice").sort('start', 1):
        for i in range(0, len(temporalSlice['topics'])):
            # Calcola un dizionario con ogni parola di un topic e la sua frequenza
            frequencies = {word['word'] : word['prob'] for word in temporalSlice['topics'][i]}

            # Genera la wordcloud
            wordcloud = WordCloud(width=800, height=800, background_color ='white', min_font_size = 8)
            wordcloud.generate_from_frequencies(frequencies=frequencies)

            plt.figure(figsize=(10, 10), facecolor=None)
            plt.imshow(wordcloud)
            plt.axis('off')
            plt.title(f'Topic {i+1}', fontsize=30, pad=30)
            plt.savefig(f"../../Resources/WordClouds/topic_{i+1}_{temporalSlice['start']}_{temporalSlice['end']}.png")
            plt.close()

            if i % 12 == 0:
                pdf.add_page()
            if i == 0:
                pdf.cell(0, txt=f"{temporalSlice['start']} - {temporalSlice['end']}", align='C');

            pdf.image(f"../../Resources/WordClouds/topic_{i + 1}_{temporalSlice['start']}_{temporalSlice['end']}.png",
                      x=70 * (i % 3), y=(math.floor((i % 12) / 3) * 70) + 15, w=70)
            os.remove(f"../../Resources/WordClouds/topic_{i+1}_{temporalSlice['start']}_{temporalSlice['end']}.png")

    pdf.output(f"../../Resources/WordClouds/topics.pdf", "F")