
![alouette satellite](Alouette-1.jpg)


- [En Français](#logiciel-de-traitement-de-lensemble-des-donn%C3%A9es-alouette-i)

# Alouette-1 - Ionogram Data Extraction - Data from Canada's First Satellite Over 60 Years In the Making

Alouette-1 is the swept-frequency topside sounder experiment that initiated Canada’s participation in space. At a time when satellites were designed and expected  to last only a few months, Alouette-1 transmitted data over 10 years to a growing international network of telemetry stations from 1962 – 72. Over this period of time the telemetry data were processed over thousands of hours by the DRTE data processing facility  at Shirley’s Bay, Ottawa – leading to hundreds of scientific publications in its time, and a family of ionospheric satellites (Alouette-2, Explorer-31, ISIS-1 and ISIS-2).

> The efforts of the unusually competent and dedicated members of the Canadian team…led to Canada’s spectacular entry into the space age with Alouette 1 on September 29, 1962.

Yet, decades later the data was nearly lost, if not for the foresight and effort to save it. Now, the data has been digitized, processed and extracted, and could be used with today’s computational methods to produce a more comprehensive model of Earth’s topside ionosphere in the 1960s - or anything else!

## How to Get Started
Read [Alouette-1 – Ionogram Data Extraction Methodology](Alouette-1 Ionogram Data Extraction Methodology-latest_ver.pdf)


# Alouette-1 - Extraction de données d'ionogrammes - Données du premier satellite canadien en développement depuis plus de 60 ans





















# Logiciel de traitement de l'ensemble des données Alouette-I
## Contexte
Le satellite [Alouette-I](https://www.asc-csa.gc.ca/fra/satellites/alouette.asp)  a été le premier satellite canadien lancé dans l'espace. Le but de son expérience principale était de comprendre la structure de la haute ionosphère.

Les données ont été enregistrées sur un film de 35 mm. L'ASC a numérisé 884952 (44,25\%) des 2 000 000 d'images d'ionogrammes brutes estimées.

 ![ionogram](ionogram.png)

Ce code sert à extraire les données et les métadonnées des ionogrammes numérisés.

## Démarrage rapide
```python
pip install -r pip-requirements.txt
cd scan2data
run user_input.py 
```

## Navigation

 - /docs contient la documentation pertinente sur les sous-modules et les sous-ensembles de scan2data
	 - La documentation peut être consultée dans un navigateur en cliquant sur /docs/_build/html/index
 - /scan2data est un progiciel basé sur Python3 qui permet de transformer les scans bruts en informations
	 - process_directory.py contient du code pour parcourir l'ensemble du processus de traitement d'un répertoire
	 - user_input.py est un moyen d'exécuter rapidement le code
 - /output-analysis contient des scripts R pour analyser et visualiser les sorties générées en CSV de /scan2data
 - /pickle stocke la trame de données finale générée par le code process_directory avec l'étiquetage du répertoire et du sous-répertoire associés
 - /feature_detection utilise le fichier pickle stocké dans /pickle pour effectuer l'extraction de traces, le nettoyage et pour extraire les paramètres d'intérêt
 - /saved_results contient les sorties d'images de feature_detection.py

## Détection de caractéristiques
Pour exécuter le code de détection des caractéristiques :
1) Exécutez /scan2data/process_directory.py sur le répertoire d'ionogrammes souhaité (spécifié dans "main")
2) Dans /feature_detection/feature_detection_main.py, indiquez le nom de fichier du fichier .pkl généré à l'étape précédente (enregistré dans /pickle)
3) Exécutez le fichier feature_detection_main.py. Les images de sortie seront sauvegardées dans /saved_results. Notez que pour le traitement à grande échelle des ionogrammes, il peut être préférable de ne pas enregistrer les images afin d'accélérer le traitement.

La détection des caractéristiques fonctionne en quelques étapes. La sortie de process_directory contient toutes les coordonnées auxquelles un point est détecté. Afin de rendre la trace plus facile à analyser scientifiquement, elle est rééchantillonnée, filtrée et lissée. Diverses méthodes peuvent être utilisées sur la trace nettoyée pour tenter d'extraire des points de données d'intérêt majeur, bien que cela nécessite un travail supplémentaire pour être généralisable.

### Conventions d'appellation des modules dans les sous-ensembles
- les modules commençant par test_ sont des modules permettant de tester la fonctionnalité du sous-paquet
- les modules commençant par draft_ sont des codes de départ qui nécessitent encore beaucoup de travail
 
## Pipeline de traitement
- Les scans bruts ont été traités par sous-répertoire
	- Chaque image brute dans le sous-répertoire a été segmentée en son ionogramme brut et en métadonnées ajustées (voir /scan2data/image_segmentation)
	- S'il était déterminé que les métadonnées se trouvaient à gauche des ionogrammes, les métadonnées de tous les ionogrammes étaient traduites en informations (voir /scan2data/metadata_translation)
		- La grille de métadonnées du point de gauche, qui fait correspondre les coordonnées des pixels de l'image des métadonnées du point de gauche au numéro, à la catégorie, a été déterminée à partir de toutes les métadonnées du point
		- La grille de métadonnées du chiffre de gauche, qui associe les coordonnées des pixels de l'image des métadonnées du chiffre de gauche au numéro, à la catégorie, a été déterminée à partir de toutes les métadonnées du chiffre
	- À partir de tous les ionogrammes extraits, on a déterminé la grille d'ionogrammes, qui permet de cartographier les coordonnées des pixels de l'image en Hz, en km (voir /scan2data/ionogram_grid_determination)
	- La trace de l'ionogramme (noir) a été extraite et mise en correspondance avec des valeurs (Hz, km) (voir /scan2data/ionogram_content_extraction)
	- Les paramètres sélectionnés ont été extraits, c'est-à-dire fmin (voir /scan2data/ionogram_content_extraction



# Software to process the Alouette-I dataset

## Background
The [Alouette-I satellite](https://www.asc-csa.gc.ca/eng/satellites/alouette.asp) was the first Canadian satellite launched into space. The goal of its main experiment was to understand the structure of the upper ionosphere. The data was recorded on 35-mm film. The CSA has scanned 884952 (44.25\%) of  the 2 000 000 estimated raw images.

![ionogram](ionogram.png)

This code is an effort to extract data and metadata from the scanned ionogram images.

## Quick start
```python
pip install -r pip-requirements.txt
cd scan2data
run user_input.py
```

## Navigation

 - /docs contains relevant documentation on the submodules and the subpackages of scan2data
	 - The documentation can be viewed in a browser by clicking on /docs/_build/html/index
 - /scan2data is a Python3-based package to transform raw scans into information
	 - process_directory.py contains code to through the entire processing pipeline for a directory
	 - user_input.py is a way to quickly run the code
 - /output-analysis contains R scripts to analyze and visualize CSV-generated outputs of /scan2data
 - /pickle stores the final dataframe generated by the process_directory code with the associated directory and subdirectory labeling
 - /feature_detection uses the pickle file stored in /pickle to perform trace extraction, cleaning, and to extract parameters of interest
 - /saved_results contains the image outputs from feature_detection.py

## Feature detection
To run the feature detection code:
1) Run /scan2data/process_directory.py on the desired ionograms directory (specified in 'main')
2) In /feature_detection/feature_detection_main.py, specify the file_name of the .pkl file generated in the previous step (saved in /pickle)
3) Run feature_detection_main.py. Output images will be saved to /saved_results. Note that for large-scale processing of ionograms, it may be preferable not to save the images in order to speed up the processing.

The feature detection works using a few steps. The output from process_directory contains all the coordinates at which a point is detected. In order to make the trace more amenable to scientific analysis, it is resampled, filtered, and smoothed. Various methods can be used on the cleaned trace to attempt to extract data points of key interest, although this needs further work to be generalizable.

### Naming conventions for modules in subpackages
- modules starting with test_ are modules to test the subpackage's functionality
- modules starting with draft_ are starting code that still needs a lot of work
 
## Processing pipeline
- The raw scans were processed by subdirectory
	- Each raw image in the subdirectory was segmented into its raw ionogram and trimmed metadata (see /scan2data/image_segmentation)
	- If the metadata was determined to be located on the left of ionograms, the metadata of all the ionograms was translated into information (see /scan2data/metadata_translation)
		- The leftside dot metadata grid, which maps image pixel coordinates of leftside dot metadata to number, category, was determined  from all the dot metadata
		- The leftside digit metadata grid, which maps image pixel coordinates of leftside digit metadata to number, category, was determined  from all the digit metadata
	- From all the extracted ionograms, the ionogram grid, which maps image pixel coordinates to Hz, km mappings,  was determined (see /scan2data/ionogram_grid_determination)
	- The ionogram trace (black) was extracted and mapped to (Hz, km) values (see /scan2data/ionogram_content_extraction)
	- Select parameters were extracted i.e. fmin (see /scan2data/ionogram_content_extraction


## Créateur | Creators
Jenisha Patel  
Etienne Low-Décarie  
Wasiq Mohammad  









