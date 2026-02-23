# Final Project

### Main Links:

This is the link to my project on Overleaf: [link](https://www.overleaf.com/read/jgsvsjqsrsyn#a5ef30). 

This is the link to my Zotero Group library: [link](https://www.zotero.org/groups/6438794/navy).

Main Data set from [link](https://datacatalog.worldbank.org/search/dataset/0037580/global-shipping-traffic-density) under Global Ship Density zip that was Last Updated: May 3, 2021 With Size: 510.1 MB



---

# Maritime Security as a Global Public Good

## Trade Frictions, Chokepoint Vulnerability, and a Scenario Engine Using AIS Shipping Density

**Author:** Carlos Carp
**Course:** MACS 30200
**Institution:** University of Chicago
**Date:** February 2026

---

## 1. Project Description

### Abstract

This project quantifies the economic value of maritime security provision by modeling global shipping disruptions at key chokepoints. Approximately 80% of global trade by volume is transported by sea, and a large share of this trade passes through a small number of physical bottlenecks (e.g., Suez Canal, Panama Canal, Strait of Malacca).

Using global AIS (Automatic Identification System) ship-density raster data from the IMF World Seaborne Trade Monitoring System, I construct:

* A 78-node global maritime transport network
* Endogenous rerouting under disruption
* Congestion-sensitive edge costs
* Counterfactual scenario simulations

The framework estimates how chokepoint closures, partial degradation, or increased security risk premia affect:

* Shortest-path distances
* Network rerouting patterns
* Port-level vulnerability
* Daily rerouting cost estimates

---

## 2. Main Findings

### Full Closure Impacts (Illustrative Results)

* **Panama Canal closure** → mean rerouting: 17,840 km
* **Suez Canal closure** → mean rerouting: 9,949 km
* **Gibraltar closure** → mean rerouting: 10,624 km
* No OD pairs disconnected due to modeled bypass routes

### Estimated Daily Rerouting Costs

| Chokepoint | Estimated Cost per Day |
| ---------- | ---------------------- |
| Gibraltar  | $116M–$233M            |
| Panama     | $33M–$66M              |
| Suez       | $27M–$53M              |
| Malacca    | $20M–$41M              |

Aggregate daily exposure across chokepoints:
**$230M–$460M/day**

This avoided cost is conceptualized as the **“hegemonic dividend”** from maritime security provision.

---

## 3. Repository Structure

```
.
├── MainPaper_Draft20260215.pdf
├── Presentation.pdf
├── _Staging&code/Scripts/ # All the scripts used, but run scripts from the folder level above, see below
├── _Staging&code/config.yaml # Edit as needed to make your netowrk. This now defines a 78-node network ~155 edges
├── _Staging&code/01_index_readings.py # this is not used here, but the orignal project had many issues with references, so code was needed to generate in-situ bib
├── _Staging&code/02_raster_quicklook.py # Due to massive GeoTIFF (shipdensity_global.tif), this loads very little to memory and makes some plots
├── _Staging&code/03_chokepoint_stats.py # Computes intensity around key chokepoints by windowed reads on the massive raster.
├── _Staging&code/04_scenario_engine.py # Maritime transport network scenario engine (78-node expanded network). It reads node/edge definitions from config.yaml. Computes all port-to-port shortest paths & runs full-closure scenarios for each of the 7 chokepoints.
├── _Staging&code/04b_enhanced_scenarios.py # Enhanced maritime network scenario engine (78-node network) - generates day-cost of chokepoint closure estimates and other derived estimates. 
├── _Staging&code/07_map_visualizations.py # Generate all 10 map-based figures for the main paper body.
├── _Staging&code/08_appendix_figures.py # Generate all appendix figures (7 for A, 6 for B and 6 for C).
├── _Staging&code/Data/                # (not included - too large, but download the 500mb zip file and unpack to 9gb here)
├── _Staging&code/Figures/generated/   # Output plots
├── _Staging&code/Tables/generated/   # Output tables
├── requirements.txt
└── README.md
```

---

## 4. Data

### AIS Shipping Density Raster

* Source: IMF World Seaborne Trade Monitoring System
* Period: Jan 2015 – Feb 2021
* Resolution: 0.005° × 0.005°
* File size: ~9GB GeoTIFF (+ pyramids)

**Note:** Raw AIS raster data is not included due to size constraints.
Users must download from World_Bank-IMF from [link](https://datacatalog.worldbank.org/search/dataset/0037580/global-shipping-traffic-density) under Global Ship Density zip that was Last Updated: May 3, 2021 With Size: 510.1 MB

---

## 5. Requirements

### Python Version

Developed using:

```
Python 3.11.6
```

To check your version:

```bash
python --version
```

---

### Required Python Packages

Install using:

```bash
pip install -r requirements.txt
```

### requirements.txt

```
numpy==1.26.4
pandas==2.2.0
matplotlib==3.8.2
seaborn==0.13.2
networkx==3.2.1
scipy==1.11.4
pyyaml==6.0.1
rasterio==1.3.9
geopandas==0.14.3
shapely==2.0.3
tqdm==4.66.2
```

If using conda:

```bash
conda create -n maritime python=3.11
conda activate maritime
pip install -r requirements.txt
```

---

## 6. Configuration

All file paths are hard-coded to the current structure starting as root from _Staging&code/

The network and chokepoint parameters are defined in:

```
config.yaml
```

In the future we will add ways to edit this file to:

* Set raster file location
* Modify chokepoint bounding boxes
* Adjust congestion elasticity (λ)
* Adjust security risk parameters

---

## 7. Reproducing Results

After downloading data ans saving it to the Data folder insdie the staging area - i.e. getting the data from [link](https://datacatalog.worldbank.org/search/dataset/0037580/global-shipping-traffic-density) under Global Ship Density zip that was Last Updated: May 3, 2021 With Size: 510.1 MB and unpacking the data into the _Staging&code/Data/    

### Step 1 — Raster Inspection

Generate global quicklook plot:

```bash
python 02_raster_quicklook.py
```

Output:

```
- Figures/generated/global_quicklook.png
- Tables/generated/raster_summary.csv
```

---

### Step 2 — Chokepoint Statistics

Compute AIS density statistics for bounding boxes:

```bash
python 03_chokepoint_stats.py
```

Outputs:
```
- Tables/generated/chokepoint_intensity.csv
- Figures/generated/chokepoint_ranking.png
```

* Chokepoint ranking figure
* Suez descriptive statistics
* Density summary plots

---

### Step 3 — Baseline Network Construction

Construct maritime network and baseline shortest paths:

```bash
python 04_scenario_engine.py
```

```
Outputs:
- Tables/generated/scenario_deltas.csv
- Tables/generated/scenario_summary_stats.csv
- Figures/generated/scenario_summary.png
- Figures/generated/scenario_example.png
```
"""
* OD shortest path matrix
* Baseline cost summary
* Network visualization

---

### Step 4 — Scenario Simulations

Run enhanced disruption simulations:

```bash
python 04b_enhanced_scenarios.py
```
```
Outputs:
- Tables/generated/partial_degradation_results.csv
- Figures/generated/partial_degradation_surface.png
- Tables/generated/port_vulnerability.csv
- Figures/generated/port_vulnerability_heatmap.png
- Tables/generated/security_scenario_results.csv
- Figures/generated/security_scenario_comparison.png
- Tables/generated/chokepoint_density_table.tex
- Tables/generated/scenario_deltas_table.tex
- Tables/generated/cost_per_day_table.tex
```


Scenarios:

* Full chokepoint closure
* Partial degradation (α multipliers)
* Security risk premium shocks

Outputs:

* Mean rerouting distances
* % affected OD pairs
* Cost summaries

---

### Step 5 — Map Visualizations

Generate maps and heatmaps:

```bash
python 07_map_visualizations.py
python 08_appendix_figures.py
```

Outputs:
```
Outputs from 07_map_visualizations.py (in Figures/generated/):
  - bounding_box_example.png
  - network_world_map.png
  - edge_cost_diagram.png
  - security_shifter_comparison.png
  - congestion_calibration.png
  - route_choice_before_after.png
  - suez_scenario_3panel.png
  - suez_intensity_detail.png
  - suez_density_stats.png
  - closure_impact_map.png
```

```
Outputs from 08_Appendix_figures.py (in Figures/generated/):
Appendix A: Chokepoint Bounding Box Maps
  - appendix_all_bboxes_world.png     (world map with all 6 bounding boxes)
  - appendix_bbox_[name].png x 6      (individual zoomed maps)

Appendix B: Chokepoint Traffic Density Close-ups
  - appendix_density_[name].png x 6   (AIS density overlay for each chokepoint)

Appendix C: Chokepoint Descriptive Statistics
  - appendix_stats_[name].png x 6     (4-panel stats for each chokepoint)
```

* Port vulnerability heatmaps
* Global closure maps
* Appendix bounding box figures

---

## 8. Reproducibility Notes

* All results depend on AIS raster density inputs.
* Random seeds are fixed where applicable.
* No stochastic elements affect deterministic shortest-path results.
* All figures in the presentation and paper can be regenerated from scripts in numerical order.

Week 9 peer reviewers should be able to:

1. Install environment and download the data from [link](https://datacatalog.worldbank.org/search/dataset/0037580/global-shipping-traffic-density) under Global Ship Density zip that was Last Updated: May 3, 2021 With Size: 510.1 MB and unpacking the data into the _Staging&code/Data/ 
2. Adjust `config.yaml` # not needed but here for throughness 
3. Run scripts in order
4. Reproduce tables and figures from the paper

---

## 9. Limitations for Replication

* Raw AIS raster not included due to file size - AGAIN, you need to downlaod the data from [link](https://datacatalog.worldbank.org/search/dataset/0037580/global-shipping-traffic-density) under Global Ship Density zip that was Last Updated: May 3, 2021 With Size: 510.1 MB and unpacking the data into the _Staging&code/Data/ 
* Security risk premia are scenario parameters (not empirically calibrated)
* Network is stylized (78 nodes) rather than fully global port coverage

---

## 10. How to Cite This Repository

If referencing this work:

**APA style:**

Carpi, C. (2026). *Maritime Security as a Global Public Good: Trade Frictions and Chokepoint Vulnerability* [Code repository]. University of Chicago.

**BibTeX:**

```bibtex
@misc{carpi2026maritime,
  author = {Carpi, Carlos},
  title = {Maritime Security as a Global Public Good: Trade Frictions and Chokepoint Vulnerability},
  year = {2026},
  note = {Winter 2026's MACS 30200, University of Chicago}
}
```

---

## 11. Contact


Author: Carlos Carpi  
Email: carpi@uchicago.edu  
University of Chicago, MACS 30200 for Winter 2026
