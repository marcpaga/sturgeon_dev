import os

from sturgeon_dev.utils import load_labels_encoding


# methylation encoding
METHYLATION_ENCODING = {
    0: -1,
    1:  1
}
NO_MEASURED = 0

NUM_PROBES = 428643

# label encoding

DIAGNOSTICS_ENCODING, DIAGNOSTICS_DECODING = load_labels_encoding(
    os.path.join(
        os.path.dirname(__file__), 
        '../static/diagnostics.json'
    )
)


CLASS_COLORS = {
    'Control - CONTR - ADENOPIT':'#c9cbcc',
    'Control - CONTR - CEBM':'#b4b6b9',
    'Control - CONTR - HEMI':'#8e8f93',
    'Control - CONTR - HYPTHAL':'#7a7e80',
    'Control - CONTR - INFLAM':'#d3d3d3',
    'Control - CONTR - PINEAL':'#68696b',
    'Control - CONTR - PONS':'#535456',
    'Control - CONTR - REACT':'#ceced0',
    'Control - CONTR - WM':'#9fa1a4',
    'Embryonal - ATRT - MYC':'#4473b1',
    'Embryonal - ATRT - SHH':'#5d95cf',
    'Embryonal - ATRT - TYR':'#73a2d7',
    'Embryonal - CNS NB - FOXR2':'#6488c6',
    'Embryonal - ETMR - ETMR': '#5992ce',
    'Embryonal - HGNET - BCOR':'#7695c6',
    'Embryonal - MB G3G4 - G3':'#4c82c0',
    'Embryonal - MB G3G4 - G4':'#9cb8e1',
    'Embryonal - MB SHH - CHL AD':'#567fbf',
    'Embryonal - MB SHH - INF':'#92b4dd',
    'Embryonal - MB WNT - WNT':'#779fd4',
    'Ependymal - EPN - MPE':'#9e2543',
    'Ependymal - EPN - PF A':'#c62655',
    'Ependymal - EPN - PF B':'#c9465d',
    'Ependymal - EPN - RELA':'#c15958',
    'Ependymal - EPN - SPINE':'#ce335c',
    'Ependymal - EPN - YAP':'#d36073',
    'Ependymal - SUBEPN - PF':'#c4293c',
    'Ependymal - SUBEPN - SPINE':'#d04f53',
    'Ependymal - SUBEPN - ST':'#cf5165',
    'Glio-neuronal - CN - CN':'#a05c2f',
    'Glio-neuronal - DLGNT - DLGNT':'#ba7548',
    'Glio-neuronal - ENB - A':'#91522c',
    'Glio-neuronal - ENB - B':'#db814e',
    'Glio-neuronal - LGG - DIG/DIA':'#b5592f',
    'Glio-neuronal - LGG - DNT':'#b35c33',
    'Glio-neuronal - LGG - GG':'#d27448',
    'Glio-neuronal - LGG - RGNT':'#df8f52',
    'Glio-neuronal - LIPN - LIPN':'#c67c41',
    'Glio-neuronal - PGG - nC':'#dc8139',
    'Glio-neuronal - RETB - RETB':'#e19262',
    'Glioblastoma - DMG - K27':'#abcf8c',
    'Glioblastoma - GBM - G34':'#7eba6d',
    'Glioblastoma - GBM - MES':'#69874b',
    'Glioblastoma - GBM - MID':'#538043',
    'Glioblastoma - GBM - MYCN':'#85bb5c',
    'Glioblastoma - GBM - RTK I':'#87bd59',
    'Glioblastoma - GBM - RTK II':'#9ac474',
    'Glioblastoma - GBM - RTK III':'#94c054',
    'Glioma IDH - A IDH - A IDH':'#dabb3b',
    'Glioma IDH - A IDH - HG':'#d1c946',
    'Glioma IDH - O IDH - O IDH':'#f4e357',
    'Haematopoeitic - LYMPHO - LYMPHO':'#381b4a',
    'Haematopoeitic - PLASMA - PLASMA':'#5a396b',
    'Melanocytic - MELAN - MELAN':'#33496c',
    'Melanocytic - MELCYT - MELCYT':'#232d4a',
    'Mesenchymal - CHORDM - CHORDM':'#b88dbc',
    'Mesenchymal - EFT - CIC':'#8560a5',
    'Mesenchymal - EWS - EWS':'#9571ae',
    'Mesenchymal - HMB - HMB':'#a06fad',
    'Mesenchymal - MNG - MNG':'#9f6fad',
    'Mesenchymal - SFT HMPC - SFT HMPC':'#b17bb2',
    'Nerve - SCHW - SCHW':'#f5d9ad',
    'Nerve - SCHW - SCHW MEL':'#ecc58a',
    'Other glioma - ANA PA - ANA PA':'#6555a2',
    'Other glioma - CHGL - CHGL':'#8f7cb7',
    'Other glioma - HGNET - MN1':'#998dc0',
    'Other glioma - IHG - IHG':'#bfb0d1',
    'Other glioma - LGG MYB - MYB':'#7262a8',
    'Other glioma - LGG PA - PA MID':'#9a88bb',
    'Other glioma - LGG PA - PA PF':'#8673b2',
    'Other glioma - LGG PA - PA/GG ST':'#6e5798',
    'Other glioma - LGG SEGA - SEGA':'#8472b1',
    'Other glioma - PXA - PXA':'#6454a1',
    'Pineal - PIN T - PB A':'#c7e0ab',
    'Pineal - PIN T - PB B':'#b5d299',
    'Pineal - PIN T - PPT':'#b7d397',
    'Pineal - PTPR - A':'#b1d195',
    'Pineal - PTPR - B':'#c1d998',
    'Plexus - PLEX - AD':'#664522',
    'Plexus - PLEX - PED A':'#875a31',
    'Plexus - PLEX - PED B':'#624530',
    'Sella - CPH - ADM':'#90c7c1',
    'Sella - CPH - PAP':'#81c3ca',
    'Sella - PITAD ACTH - ACTH':'#50807b',
    'Sella - PITAD FSH LH - FSH LH':'#6ba69d',
    'Sella - PITAD PRL - PRL':'#68b7af',
    'Sella - PITAD STH - STH DNS A':'#78beb8',
    'Sella - PITAD STH - STH DNS B':'#6fbdc0',
    'Sella - PITAD STH - STH SPA':'#7dc1b7',
    'Sella - PITAD TSH - TSH':'#77beb7',
    'Sella - PITUI SCO GCT - SCO GCT':'#86c8ce',
}