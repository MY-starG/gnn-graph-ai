<p align="center">
    <img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" align="center" width="30%">
</p>
<p align="center"><h1 align="center">GNN-GRAPH-AI</h1></p>
<p align="center">
	<em><code>❯ REPLACE-ME</code></em>
</p>
<p align="center">
	<img src="https://img.shields.io/github/license/sanu123-mj/gnn-graph-ai?style=default&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/sanu123-mj/gnn-graph-ai?style=default&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/sanu123-mj/gnn-graph-ai?style=default&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/sanu123-mj/gnn-graph-ai?style=default&color=0080ff" alt="repo-language-count">
</p>
<p align="center"><!-- default option, no dependency badges. -->
</p>
<p align="center">
	<!-- default option, no dependency badges. -->
</p>
<br>

##  Table of Contents

- [ Overview](#-overview)
- [ Features](#-features)
- [ Project Structure](#-project-structure)
  - [ Project Index](#-project-index)
- [ Getting Started](#-getting-started)
  - [ Prerequisites](#-prerequisites)
  - [ Installation](#-installation)
  - [ Usage](#-usage)
  - [ Testing](#-testing)
- [ Project Roadmap](#-project-roadmap)
- [ Contributing](#-contributing)
- [ License](#-license)
- [ Acknowledgments](#-acknowledgments)

---

##  Overview

<code>❯ REPLACE-ME</code>

---

##  Features

<code>❯ REPLACE-ME</code>

---

##  Project Structure

```sh
└── gnn-graph-ai/
    ├── LICENSE
    ├── README.md
    ├── data
    │   ├── CiteSeer
    │   ├── PubMed
    │   └── cora
    ├── evaluation
    │   ├── __pycache__
    │   └── metrics.py
    ├── experiments
    │   ├── ablation_study.py
    │   ├── compare_models.py
    │   ├── evaluate_all_models.py
    │   ├── evaluate_all_models_advanced.py
    │   ├── evaluate_augmentation.py
    │   ├── hyperparameter_search.py
    │   ├── multi_run_evaluation.py
    │   └── run_explainability.py
    ├── explainability
    │   └── gnn_explainer.py
    ├── file_paths.txt
    ├── final GNN document.docx
    ├── folder_tree.txt
    ├── models
    │   ├── __pycache__
    │   ├── gat.py
    │   ├── gcn.py
    │   ├── graphsage.py
    │   └── saved
    ├── notebooks
    │   └── analysis.ipynb
    ├── requirements.txt
    ├── results
    │   ├── ablation_studies
    │   ├── augmentation_effects
    │   ├── citeseer_evaluation
    │   ├── explainability
    │   ├── explainability_demo
    │   ├── figures
    │   ├── hyperparameter_search
    │   ├── latex_tables
    │   ├── link_prediction
    │   ├── multiple_runs
    │   ├── pubmed_evaluation
    │   ├── reports
    │   └── visualizations
    ├── training
    │   ├── data
    │   ├── train_ensemble.py
    │   ├── train_gat.py
    │   ├── train_gcn.py
    │   ├── train_graphsage.py
    │   ├── train_link_prediction.py
    │   ├── train_node_classification.py
    │   └── venv
    └── utils
        ├── __pycache__
        ├── data_loader.py
        ├── graph_augmentation.py
        └── visualization.py
```


###  Project Index
<details open>
	<summary><b><code>GNN-GRAPH-AI/</code></b></summary>
	<details> <!-- __root__ Submodule -->
		<summary><b>__root__</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/folder_tree.txt'>folder_tree.txt</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/requirements.txt'>requirements.txt</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/file_paths.txt'>file_paths.txt</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- explainability Submodule -->
		<summary><b>explainability</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/explainability/gnn_explainer.py'>gnn_explainer.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- results Submodule -->
		<summary><b>results</b></summary>
		<blockquote>
			<details>
				<summary><b>hyperparameter_search</b></summary>
				<blockquote>
					<details>
						<summary><b>configs</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/results/hyperparameter_search/configs/sage_random_search_results.json'>sage_random_search_results.json</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/results/hyperparameter_search/configs/gat_random_search_results.json'>gat_random_search_results.json</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/results/hyperparameter_search/configs/best_configurations.json'>best_configurations.json</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/results/hyperparameter_search/configs/gcn_grid_search_results.json'>gcn_grid_search_results.json</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							</table>
						</blockquote>
					</details>
				</blockquote>
			</details>
			<details>
				<summary><b>explainability</b></summary>
				<blockquote>
					<details>
						<summary><b>gcn</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/results/explainability/gcn/advanced_explanation_summary.txt'>advanced_explanation_summary.txt</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							</table>
						</blockquote>
					</details>
					<details>
						<summary><b>graphsage</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/results/explainability/graphsage/advanced_explanation_summary.txt'>advanced_explanation_summary.txt</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							</table>
						</blockquote>
					</details>
				</blockquote>
			</details>
			<details>
				<summary><b>ablation_studies</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/results/ablation_studies/all_ablation_results.json'>all_ablation_results.json</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					</table>
				</blockquote>
			</details>
			<details>
				<summary><b>multiple_runs</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/results/multiple_runs/statistical_report.txt'>statistical_report.txt</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/results/multiple_runs/results_table.tex'>results_table.tex</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/results/multiple_runs/all_runs_data.json'>all_runs_data.json</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					</table>
				</blockquote>
			</details>
			<details>
				<summary><b>citeseer_evaluation</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/results/citeseer_evaluation/citeseer_results.json'>citeseer_results.json</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					</table>
					<details>
						<summary><b>reports</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/results/citeseer_evaluation/reports/citeseer_evaluation_20260104_131608.txt'>citeseer_evaluation_20260104_131608.txt</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							</table>
						</blockquote>
					</details>
				</blockquote>
			</details>
			<details>
				<summary><b>reports</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/results/reports/comprehensive_evaluation_20260103_201201.txt'>comprehensive_evaluation_20260103_201201.txt</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					</table>
				</blockquote>
			</details>
			<details>
				<summary><b>explainability_demo</b></summary>
				<blockquote>
					<details>
						<summary><b>gcn</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/results/explainability_demo/gcn/explanations_data.json'>explanations_data.json</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/results/explainability_demo/gcn/statistical_report.txt'>statistical_report.txt</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							</table>
						</blockquote>
					</details>
					<details>
						<summary><b>graphsage</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/results/explainability_demo/graphsage/explanations_data.json'>explanations_data.json</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/results/explainability_demo/graphsage/statistical_report.txt'>statistical_report.txt</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							</table>
						</blockquote>
					</details>
				</blockquote>
			</details>
			<details>
				<summary><b>latex_tables</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/results/latex_tables/comparison_table.tex'>comparison_table.tex</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					</table>
				</blockquote>
			</details>
			<details>
				<summary><b>pubmed_evaluation</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/results/pubmed_evaluation/pubmed_results.json'>pubmed_results.json</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					</table>
					<details>
						<summary><b>reports</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/results/pubmed_evaluation/reports/pubmed_evaluation_20260104_131806.txt'>pubmed_evaluation_20260104_131806.txt</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							</table>
						</blockquote>
					</details>
				</blockquote>
			</details>
		</blockquote>
	</details>
	<details> <!-- evaluation Submodule -->
		<summary><b>evaluation</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/evaluation/metrics.py'>metrics.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- experiments Submodule -->
		<summary><b>experiments</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/experiments/evaluate_all_models_advanced.py'>evaluate_all_models_advanced.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/experiments/hyperparameter_search.py'>hyperparameter_search.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/experiments/evaluate_augmentation.py'>evaluate_augmentation.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/experiments/compare_models.py'>compare_models.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/experiments/evaluate_all_models.py'>evaluate_all_models.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/experiments/run_explainability.py'>run_explainability.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/experiments/multi_run_evaluation.py'>multi_run_evaluation.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/experiments/ablation_study.py'>ablation_study.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- notebooks Submodule -->
		<summary><b>notebooks</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/notebooks/analysis.ipynb'>analysis.ipynb</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- models Submodule -->
		<summary><b>models</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/models/gat.py'>gat.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/models/gcn.py'>gcn.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/models/graphsage.py'>graphsage.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			</table>
			<details>
				<summary><b>saved</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/models/saved/gcn_best.pth'>gcn_best.pth</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/models/saved/gat_best.pth'>gat_best.pth</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/models/saved/graphsage_best.pth'>graphsage_best.pth</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					</table>
				</blockquote>
			</details>
		</blockquote>
	</details>
	<details> <!-- training Submodule -->
		<summary><b>training</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/train_ensemble.py'>train_ensemble.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/train_graphsage.py'>train_graphsage.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/train_gat.py'>train_gat.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/train_gcn.py'>train_gcn.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/train_link_prediction.py'>train_link_prediction.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/train_node_classification.py'>train_node_classification.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			</table>
			<details>
				<summary><b>venv</b></summary>
				<blockquote>
					<details>
						<summary><b>Scripts</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Scripts/activate'>activate</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Scripts/Activate.ps1'>Activate.ps1</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Scripts/deactivate.bat'>deactivate.bat</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Scripts/activate.bat'>activate.bat</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							</table>
						</blockquote>
					</details>
					<details>
						<summary><b>Lib</b></summary>
						<blockquote>
							<details>
								<summary><b>site-packages</b></summary>
								<blockquote>
									<details>
										<summary><b>pip</b></summary>
										<blockquote>
											<table>
											<tr>
												<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/py.typed'>py.typed</a></b></td>
												<td><code>❯ REPLACE-ME</code></td>
											</tr>
											<tr>
												<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/__main__.py'>__main__.py</a></b></td>
												<td><code>❯ REPLACE-ME</code></td>
											</tr>
											<tr>
												<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/__pip-runner__.py'>__pip-runner__.py</a></b></td>
												<td><code>❯ REPLACE-ME</code></td>
											</tr>
											</table>
											<details>
												<summary><b>_vendor</b></summary>
												<blockquote>
													<table>
													<tr>
														<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/vendor.txt'>vendor.txt</a></b></td>
														<td><code>❯ REPLACE-ME</code></td>
													</tr>
													<tr>
														<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/six.py'>six.py</a></b></td>
														<td><code>❯ REPLACE-ME</code></td>
													</tr>
													<tr>
														<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/typing_extensions.py'>typing_extensions.py</a></b></td>
														<td><code>❯ REPLACE-ME</code></td>
													</tr>
													</table>
													<details>
														<summary><b>distlib</b></summary>
														<blockquote>
															<table>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/distlib/compat.py'>compat.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/distlib/metadata.py'>metadata.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/distlib/locators.py'>locators.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/distlib/database.py'>database.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/distlib/version.py'>version.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/distlib/scripts.py'>scripts.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/distlib/markers.py'>markers.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/distlib/wheel.py'>wheel.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/distlib/resources.py'>resources.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/distlib/index.py'>index.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/distlib/util.py'>util.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/distlib/manifest.py'>manifest.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															</table>
														</blockquote>
													</details>
													<details>
														<summary><b>tenacity</b></summary>
														<blockquote>
															<table>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/tenacity/py.typed'>py.typed</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/tenacity/wait.py'>wait.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/tenacity/_utils.py'>_utils.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/tenacity/before_sleep.py'>before_sleep.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/tenacity/nap.py'>nap.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/tenacity/tornadoweb.py'>tornadoweb.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/tenacity/after.py'>after.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/tenacity/_asyncio.py'>_asyncio.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/tenacity/stop.py'>stop.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/tenacity/before.py'>before.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/tenacity/retry.py'>retry.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															</table>
														</blockquote>
													</details>
													<details>
														<summary><b>resolvelib</b></summary>
														<blockquote>
															<table>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/resolvelib/py.typed'>py.typed</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/resolvelib/resolvers.py'>resolvers.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/resolvelib/providers.py'>providers.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/resolvelib/structs.py'>structs.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/resolvelib/reporters.py'>reporters.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															</table>
															<details>
																<summary><b>compat</b></summary>
																<blockquote>
																	<table>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/resolvelib/compat/collections_abc.py'>collections_abc.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	</table>
																</blockquote>
															</details>
														</blockquote>
													</details>
													<details>
														<summary><b>urllib3</b></summary>
														<blockquote>
															<table>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/urllib3/connection.py'>connection.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/urllib3/_collections.py'>_collections.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/urllib3/filepost.py'>filepost.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/urllib3/response.py'>response.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/urllib3/request.py'>request.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/urllib3/fields.py'>fields.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/urllib3/connectionpool.py'>connectionpool.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/urllib3/poolmanager.py'>poolmanager.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/urllib3/exceptions.py'>exceptions.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/urllib3/_version.py'>_version.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															</table>
															<details>
																<summary><b>contrib</b></summary>
																<blockquote>
																	<table>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/urllib3/contrib/ntlmpool.py'>ntlmpool.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/urllib3/contrib/socks.py'>socks.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/urllib3/contrib/_appengine_environ.py'>_appengine_environ.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/urllib3/contrib/securetransport.py'>securetransport.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/urllib3/contrib/pyopenssl.py'>pyopenssl.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/urllib3/contrib/appengine.py'>appengine.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	</table>
																	<details>
																		<summary><b>_securetransport</b></summary>
																		<blockquote>
																			<table>
																			<tr>
																				<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/urllib3/contrib/_securetransport/low_level.py'>low_level.py</a></b></td>
																				<td><code>❯ REPLACE-ME</code></td>
																			</tr>
																			<tr>
																				<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/urllib3/contrib/_securetransport/bindings.py'>bindings.py</a></b></td>
																				<td><code>❯ REPLACE-ME</code></td>
																			</tr>
																			</table>
																		</blockquote>
																	</details>
																</blockquote>
															</details>
															<details>
																<summary><b>packages</b></summary>
																<blockquote>
																	<table>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/urllib3/packages/six.py'>six.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	</table>
																	<details>
																		<summary><b>backports</b></summary>
																		<blockquote>
																			<table>
																			<tr>
																				<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/urllib3/packages/backports/makefile.py'>makefile.py</a></b></td>
																				<td><code>❯ REPLACE-ME</code></td>
																			</tr>
																			<tr>
																				<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/urllib3/packages/backports/weakref_finalize.py'>weakref_finalize.py</a></b></td>
																				<td><code>❯ REPLACE-ME</code></td>
																			</tr>
																			</table>
																		</blockquote>
																	</details>
																</blockquote>
															</details>
															<details>
																<summary><b>util</b></summary>
																<blockquote>
																	<table>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/urllib3/util/connection.py'>connection.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/urllib3/util/wait.py'>wait.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/urllib3/util/proxy.py'>proxy.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/urllib3/util/response.py'>response.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/urllib3/util/request.py'>request.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/urllib3/util/timeout.py'>timeout.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/urllib3/util/url.py'>url.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/urllib3/util/queue.py'>queue.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/urllib3/util/ssl_.py'>ssl_.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/urllib3/util/ssl_match_hostname.py'>ssl_match_hostname.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/urllib3/util/retry.py'>retry.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/urllib3/util/ssltransport.py'>ssltransport.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	</table>
																</blockquote>
															</details>
														</blockquote>
													</details>
													<details>
														<summary><b>msgpack</b></summary>
														<blockquote>
															<table>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/msgpack/ext.py'>ext.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/msgpack/fallback.py'>fallback.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/msgpack/exceptions.py'>exceptions.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															</table>
														</blockquote>
													</details>
													<details>
														<summary><b>tomli</b></summary>
														<blockquote>
															<table>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/tomli/py.typed'>py.typed</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/tomli/_types.py'>_types.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/tomli/_re.py'>_re.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/tomli/_parser.py'>_parser.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															</table>
														</blockquote>
													</details>
													<details>
														<summary><b>platformdirs</b></summary>
														<blockquote>
															<table>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/platformdirs/windows.py'>windows.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/platformdirs/py.typed'>py.typed</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/platformdirs/unix.py'>unix.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/platformdirs/api.py'>api.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/platformdirs/__main__.py'>__main__.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/platformdirs/version.py'>version.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/platformdirs/macos.py'>macos.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/platformdirs/android.py'>android.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															</table>
														</blockquote>
													</details>
													<details>
														<summary><b>cachecontrol</b></summary>
														<blockquote>
															<table>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/cachecontrol/py.typed'>py.typed</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/cachecontrol/heuristics.py'>heuristics.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/cachecontrol/wrapper.py'>wrapper.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/cachecontrol/controller.py'>controller.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/cachecontrol/serialize.py'>serialize.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/cachecontrol/cache.py'>cache.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/cachecontrol/_cmd.py'>_cmd.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/cachecontrol/adapter.py'>adapter.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/cachecontrol/filewrapper.py'>filewrapper.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															</table>
															<details>
																<summary><b>caches</b></summary>
																<blockquote>
																	<table>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/cachecontrol/caches/redis_cache.py'>redis_cache.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/cachecontrol/caches/file_cache.py'>file_cache.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	</table>
																</blockquote>
															</details>
														</blockquote>
													</details>
													<details>
														<summary><b>distro</b></summary>
														<blockquote>
															<table>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/distro/py.typed'>py.typed</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/distro/__main__.py'>__main__.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/distro/distro.py'>distro.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															</table>
														</blockquote>
													</details>
													<details>
														<summary><b>rich</b></summary>
														<blockquote>
															<table>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/_cell_widths.py'>_cell_widths.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/segment.py'>segment.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/py.typed'>py.typed</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/_export_format.py'>_export_format.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/errors.py'>errors.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/_timer.py'>_timer.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/_spinners.py'>_spinners.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/live.py'>live.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/jupyter.py'>jupyter.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/control.py'>control.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/table.py'>table.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/logging.py'>logging.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/_null_file.py'>_null_file.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/_ratio.py'>_ratio.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/file_proxy.py'>file_proxy.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/screen.py'>screen.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/_windows.py'>_windows.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/json.py'>json.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/box.py'>box.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/_fileno.py'>_fileno.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/bar.py'>bar.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/padding.py'>padding.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/diagnose.py'>diagnose.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/_wrap.py'>_wrap.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/styled.py'>styled.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/__main__.py'>__main__.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/progress.py'>progress.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/emoji.py'>emoji.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/color.py'>color.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/pager.py'>pager.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/_emoji_codes.py'>_emoji_codes.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/region.py'>region.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/constrain.py'>constrain.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/theme.py'>theme.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/measure.py'>measure.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/ansi.py'>ansi.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/themes.py'>themes.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/repr.py'>repr.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/panel.py'>panel.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/_windows_renderer.py'>_windows_renderer.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/default_styles.py'>default_styles.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/containers.py'>containers.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/terminal_theme.py'>terminal_theme.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/tree.py'>tree.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/_pick.py'>_pick.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/markup.py'>markup.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/_stack.py'>_stack.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/pretty.py'>pretty.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/traceback.py'>traceback.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/style.py'>style.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/filesize.py'>filesize.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/scope.py'>scope.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/syntax.py'>syntax.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/color_triplet.py'>color_triplet.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/text.py'>text.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/palette.py'>palette.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/console.py'>console.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/highlighter.py'>highlighter.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/align.py'>align.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/prompt.py'>prompt.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/columns.py'>columns.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/cells.py'>cells.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/live_render.py'>live_render.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/_extension.py'>_extension.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/_emoji_replace.py'>_emoji_replace.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/rule.py'>rule.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/abc.py'>abc.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/progress_bar.py'>progress_bar.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/status.py'>status.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/protocol.py'>protocol.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/_inspect.py'>_inspect.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/layout.py'>layout.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/_palettes.py'>_palettes.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/spinner.py'>spinner.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/_log_render.py'>_log_render.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/_loop.py'>_loop.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/rich/_win32_console.py'>_win32_console.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															</table>
														</blockquote>
													</details>
													<details>
														<summary><b>chardet</b></summary>
														<blockquote>
															<table>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/chardet/py.typed'>py.typed</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/chardet/mbcsgroupprober.py'>mbcsgroupprober.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/chardet/big5freq.py'>big5freq.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/chardet/jpcntx.py'>jpcntx.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/chardet/mbcharsetprober.py'>mbcharsetprober.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/chardet/johabfreq.py'>johabfreq.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/chardet/charsetgroupprober.py'>charsetgroupprober.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/chardet/jisfreq.py'>jisfreq.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/chardet/langgreekmodel.py'>langgreekmodel.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/chardet/euctwfreq.py'>euctwfreq.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/chardet/utf8prober.py'>utf8prober.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/chardet/version.py'>version.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/chardet/universaldetector.py'>universaldetector.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/chardet/enums.py'>enums.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/chardet/escsm.py'>escsm.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/chardet/codingstatemachine.py'>codingstatemachine.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/chardet/sjisprober.py'>sjisprober.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/chardet/sbcharsetprober.py'>sbcharsetprober.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/chardet/resultdict.py'>resultdict.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/chardet/langhungarianmodel.py'>langhungarianmodel.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/chardet/hebrewprober.py'>hebrewprober.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/chardet/langturkishmodel.py'>langturkishmodel.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/chardet/gb2312freq.py'>gb2312freq.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/chardet/langbulgarianmodel.py'>langbulgarianmodel.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/chardet/chardistribution.py'>chardistribution.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/chardet/langrussianmodel.py'>langrussianmodel.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/chardet/macromanprober.py'>macromanprober.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/chardet/johabprober.py'>johabprober.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/chardet/mbcssm.py'>mbcssm.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/chardet/codingstatemachinedict.py'>codingstatemachinedict.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/chardet/sbcsgroupprober.py'>sbcsgroupprober.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/chardet/langthaimodel.py'>langthaimodel.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/chardet/gb2312prober.py'>gb2312prober.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/chardet/big5prober.py'>big5prober.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/chardet/euctwprober.py'>euctwprober.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/chardet/eucjpprober.py'>eucjpprober.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/chardet/cp949prober.py'>cp949prober.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/chardet/charsetprober.py'>charsetprober.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/chardet/utf1632prober.py'>utf1632prober.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/chardet/langhebrewmodel.py'>langhebrewmodel.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/chardet/latin1prober.py'>latin1prober.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/chardet/euckrprober.py'>euckrprober.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/chardet/escprober.py'>escprober.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/chardet/euckrfreq.py'>euckrfreq.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															</table>
															<details>
																<summary><b>cli</b></summary>
																<blockquote>
																	<table>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/chardet/cli/chardetect.py'>chardetect.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	</table>
																</blockquote>
															</details>
															<details>
																<summary><b>metadata</b></summary>
																<blockquote>
																	<table>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/chardet/metadata/languages.py'>languages.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	</table>
																</blockquote>
															</details>
														</blockquote>
													</details>
													<details>
														<summary><b>certifi</b></summary>
														<blockquote>
															<table>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/certifi/py.typed'>py.typed</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/certifi/__main__.py'>__main__.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/certifi/core.py'>core.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															</table>
														</blockquote>
													</details>
													<details>
														<summary><b>requests</b></summary>
														<blockquote>
															<table>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/requests/api.py'>api.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/requests/compat.py'>compat.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/requests/status_codes.py'>status_codes.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/requests/sessions.py'>sessions.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/requests/structures.py'>structures.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/requests/utils.py'>utils.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/requests/adapters.py'>adapters.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/requests/packages.py'>packages.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/requests/certs.py'>certs.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/requests/cookies.py'>cookies.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/requests/__version__.py'>__version__.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/requests/hooks.py'>hooks.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/requests/_internal_utils.py'>_internal_utils.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/requests/help.py'>help.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/requests/exceptions.py'>exceptions.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/requests/auth.py'>auth.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/requests/models.py'>models.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															</table>
														</blockquote>
													</details>
													<details>
														<summary><b>pyproject_hooks</b></summary>
														<blockquote>
															<table>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/pyproject_hooks/_compat.py'>_compat.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/pyproject_hooks/_impl.py'>_impl.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															</table>
															<details>
																<summary><b>_in_process</b></summary>
																<blockquote>
																	<table>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py'>_in_process.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	</table>
																</blockquote>
															</details>
														</blockquote>
													</details>
													<details>
														<summary><b>idna</b></summary>
														<blockquote>
															<table>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/idna/py.typed'>py.typed</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/idna/package_data.py'>package_data.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/idna/compat.py'>compat.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/idna/idnadata.py'>idnadata.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/idna/uts46data.py'>uts46data.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/idna/intranges.py'>intranges.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/idna/core.py'>core.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/idna/codec.py'>codec.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															</table>
														</blockquote>
													</details>
													<details>
														<summary><b>colorama</b></summary>
														<blockquote>
															<table>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/colorama/win32.py'>win32.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/colorama/ansi.py'>ansi.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/colorama/winterm.py'>winterm.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/colorama/ansitowin32.py'>ansitowin32.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/colorama/initialise.py'>initialise.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															</table>
														</blockquote>
													</details>
													<details>
														<summary><b>packaging</b></summary>
														<blockquote>
															<table>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/packaging/py.typed'>py.typed</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/packaging/__about__.py'>__about__.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/packaging/utils.py'>utils.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/packaging/version.py'>version.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/packaging/markers.py'>markers.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/packaging/specifiers.py'>specifiers.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/packaging/_manylinux.py'>_manylinux.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/packaging/_structures.py'>_structures.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/packaging/_musllinux.py'>_musllinux.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/packaging/requirements.py'>requirements.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/packaging/tags.py'>tags.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															</table>
														</blockquote>
													</details>
													<details>
														<summary><b>pygments</b></summary>
														<blockquote>
															<table>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/pygments/token.py'>token.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/pygments/formatter.py'>formatter.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/pygments/filter.py'>filter.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/pygments/regexopt.py'>regexopt.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/pygments/scanner.py'>scanner.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/pygments/__main__.py'>__main__.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/pygments/sphinxext.py'>sphinxext.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/pygments/style.py'>style.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/pygments/unistring.py'>unistring.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/pygments/util.py'>util.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/pygments/lexer.py'>lexer.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/pygments/console.py'>console.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/pygments/cmdline.py'>cmdline.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/pygments/modeline.py'>modeline.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/pygments/plugin.py'>plugin.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															</table>
															<details>
																<summary><b>lexers</b></summary>
																<blockquote>
																	<table>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/pygments/lexers/_mapping.py'>_mapping.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/pygments/lexers/python.py'>python.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	</table>
																</blockquote>
															</details>
															<details>
																<summary><b>formatters</b></summary>
																<blockquote>
																	<table>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/pygments/formatters/html.py'>html.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/pygments/formatters/irc.py'>irc.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/pygments/formatters/groff.py'>groff.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/pygments/formatters/bbcode.py'>bbcode.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/pygments/formatters/other.py'>other.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/pygments/formatters/pangomarkup.py'>pangomarkup.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/pygments/formatters/latex.py'>latex.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/pygments/formatters/img.py'>img.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/pygments/formatters/terminal.py'>terminal.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/pygments/formatters/rtf.py'>rtf.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/pygments/formatters/svg.py'>svg.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/pygments/formatters/_mapping.py'>_mapping.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/pygments/formatters/terminal256.py'>terminal256.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	</table>
																</blockquote>
															</details>
														</blockquote>
													</details>
													<details>
														<summary><b>pyparsing</b></summary>
														<blockquote>
															<table>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/pyparsing/results.py'>results.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/pyparsing/py.typed'>py.typed</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/pyparsing/testing.py'>testing.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/pyparsing/helpers.py'>helpers.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/pyparsing/actions.py'>actions.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/pyparsing/util.py'>util.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/pyparsing/common.py'>common.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/pyparsing/core.py'>core.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/pyparsing/exceptions.py'>exceptions.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/pyparsing/unicode.py'>unicode.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															</table>
														</blockquote>
													</details>
													<details>
														<summary><b>truststore</b></summary>
														<blockquote>
															<table>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/truststore/py.typed'>py.typed</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/truststore/_windows.py'>_windows.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/truststore/_api.py'>_api.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/truststore/_macos.py'>_macos.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/truststore/_openssl.py'>_openssl.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/truststore/_ssl_constants.py'>_ssl_constants.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															</table>
														</blockquote>
													</details>
													<details>
														<summary><b>webencodings</b></summary>
														<blockquote>
															<table>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/webencodings/tests.py'>tests.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/webencodings/labels.py'>labels.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/webencodings/x_user_defined.py'>x_user_defined.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_vendor/webencodings/mklabels.py'>mklabels.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															</table>
														</blockquote>
													</details>
												</blockquote>
											</details>
											<details>
												<summary><b>_internal</b></summary>
												<blockquote>
													<table>
													<tr>
														<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/self_outdated_check.py'>self_outdated_check.py</a></b></td>
														<td><code>❯ REPLACE-ME</code></td>
													</tr>
													<tr>
														<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/build_env.py'>build_env.py</a></b></td>
														<td><code>❯ REPLACE-ME</code></td>
													</tr>
													<tr>
														<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/pyproject.py'>pyproject.py</a></b></td>
														<td><code>❯ REPLACE-ME</code></td>
													</tr>
													<tr>
														<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/wheel_builder.py'>wheel_builder.py</a></b></td>
														<td><code>❯ REPLACE-ME</code></td>
													</tr>
													<tr>
														<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/main.py'>main.py</a></b></td>
														<td><code>❯ REPLACE-ME</code></td>
													</tr>
													<tr>
														<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/cache.py'>cache.py</a></b></td>
														<td><code>❯ REPLACE-ME</code></td>
													</tr>
													<tr>
														<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/exceptions.py'>exceptions.py</a></b></td>
														<td><code>❯ REPLACE-ME</code></td>
													</tr>
													<tr>
														<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/configuration.py'>configuration.py</a></b></td>
														<td><code>❯ REPLACE-ME</code></td>
													</tr>
													</table>
													<details>
														<summary><b>network</b></summary>
														<blockquote>
															<table>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/network/utils.py'>utils.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/network/cache.py'>cache.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/network/lazy_wheel.py'>lazy_wheel.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/network/session.py'>session.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/network/xmlrpc.py'>xmlrpc.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/network/download.py'>download.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/network/auth.py'>auth.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															</table>
														</blockquote>
													</details>
													<details>
														<summary><b>distributions</b></summary>
														<blockquote>
															<table>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/distributions/base.py'>base.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/distributions/sdist.py'>sdist.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/distributions/wheel.py'>wheel.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/distributions/installed.py'>installed.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															</table>
														</blockquote>
													</details>
													<details>
														<summary><b>cli</b></summary>
														<blockquote>
															<table>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/cli/status_codes.py'>status_codes.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/cli/parser.py'>parser.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/cli/cmdoptions.py'>cmdoptions.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/cli/main_parser.py'>main_parser.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/cli/req_command.py'>req_command.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/cli/main.py'>main.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/cli/base_command.py'>base_command.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/cli/command_context.py'>command_context.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/cli/progress_bars.py'>progress_bars.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/cli/autocompletion.py'>autocompletion.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/cli/spinners.py'>spinners.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															</table>
														</blockquote>
													</details>
													<details>
														<summary><b>resolution</b></summary>
														<blockquote>
															<table>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/resolution/base.py'>base.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															</table>
															<details>
																<summary><b>legacy</b></summary>
																<blockquote>
																	<table>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/resolution/legacy/resolver.py'>resolver.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	</table>
																</blockquote>
															</details>
															<details>
																<summary><b>resolvelib</b></summary>
																<blockquote>
																	<table>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/resolution/resolvelib/base.py'>base.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/resolution/resolvelib/found_candidates.py'>found_candidates.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/resolution/resolvelib/factory.py'>factory.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/resolution/resolvelib/resolver.py'>resolver.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/resolution/resolvelib/candidates.py'>candidates.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/resolution/resolvelib/requirements.py'>requirements.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/resolution/resolvelib/provider.py'>provider.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/resolution/resolvelib/reporter.py'>reporter.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	</table>
																</blockquote>
															</details>
														</blockquote>
													</details>
													<details>
														<summary><b>index</b></summary>
														<blockquote>
															<table>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/index/collector.py'>collector.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/index/package_finder.py'>package_finder.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/index/sources.py'>sources.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															</table>
														</blockquote>
													</details>
													<details>
														<summary><b>metadata</b></summary>
														<blockquote>
															<table>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/metadata/base.py'>base.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/metadata/pkg_resources.py'>pkg_resources.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/metadata/_json.py'>_json.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															</table>
															<details>
																<summary><b>importlib</b></summary>
																<blockquote>
																	<table>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/metadata/importlib/_dists.py'>_dists.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/metadata/importlib/_compat.py'>_compat.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/metadata/importlib/_envs.py'>_envs.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	</table>
																</blockquote>
															</details>
														</blockquote>
													</details>
													<details>
														<summary><b>models</b></summary>
														<blockquote>
															<table>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/models/scheme.py'>scheme.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/models/format_control.py'>format_control.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/models/link.py'>link.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/models/selection_prefs.py'>selection_prefs.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/models/search_scope.py'>search_scope.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/models/candidate.py'>candidate.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/models/installation_report.py'>installation_report.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/models/wheel.py'>wheel.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/models/target_python.py'>target_python.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/models/index.py'>index.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/models/direct_url.py'>direct_url.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															</table>
														</blockquote>
													</details>
													<details>
														<summary><b>operations</b></summary>
														<blockquote>
															<table>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/operations/check.py'>check.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/operations/prepare.py'>prepare.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/operations/freeze.py'>freeze.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															</table>
															<details>
																<summary><b>build</b></summary>
																<blockquote>
																	<table>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/operations/build/metadata_editable.py'>metadata_editable.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/operations/build/metadata.py'>metadata.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/operations/build/build_tracker.py'>build_tracker.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/operations/build/wheel.py'>wheel.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/operations/build/wheel_editable.py'>wheel_editable.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/operations/build/metadata_legacy.py'>metadata_legacy.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/operations/build/wheel_legacy.py'>wheel_legacy.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	</table>
																</blockquote>
															</details>
															<details>
																<summary><b>install</b></summary>
																<blockquote>
																	<table>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/operations/install/editable_legacy.py'>editable_legacy.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	<tr>
																		<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/operations/install/wheel.py'>wheel.py</a></b></td>
																		<td><code>❯ REPLACE-ME</code></td>
																	</tr>
																	</table>
																</blockquote>
															</details>
														</blockquote>
													</details>
													<details>
														<summary><b>vcs</b></summary>
														<blockquote>
															<table>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/vcs/mercurial.py'>mercurial.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/vcs/git.py'>git.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/vcs/subversion.py'>subversion.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/vcs/versioncontrol.py'>versioncontrol.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/vcs/bazaar.py'>bazaar.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															</table>
														</blockquote>
													</details>
													<details>
														<summary><b>utils</b></summary>
														<blockquote>
															<table>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/utils/encoding.py'>encoding.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/utils/compat.py'>compat.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/utils/logging.py'>logging.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/utils/direct_url_helpers.py'>direct_url_helpers.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/utils/glibc.py'>glibc.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/utils/datetime.py'>datetime.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/utils/hashes.py'>hashes.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/utils/_log.py'>_log.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/utils/wheel.py'>wheel.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/utils/filesystem.py'>filesystem.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/utils/urls.py'>urls.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/utils/compatibility_tags.py'>compatibility_tags.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/utils/misc.py'>misc.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/utils/unpacking.py'>unpacking.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/utils/entrypoints.py'>entrypoints.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/utils/_jaraco_text.py'>_jaraco_text.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/utils/packaging.py'>packaging.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/utils/setuptools_build.py'>setuptools_build.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/utils/filetypes.py'>filetypes.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/utils/temp_dir.py'>temp_dir.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/utils/subprocess.py'>subprocess.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/utils/egg_link.py'>egg_link.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/utils/models.py'>models.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/utils/virtualenv.py'>virtualenv.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/utils/deprecation.py'>deprecation.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/utils/appdirs.py'>appdirs.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															</table>
														</blockquote>
													</details>
													<details>
														<summary><b>req</b></summary>
														<blockquote>
															<table>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/req/constructors.py'>constructors.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/req/req_file.py'>req_file.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/req/req_uninstall.py'>req_uninstall.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/req/req_install.py'>req_install.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/req/req_set.py'>req_set.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															</table>
														</blockquote>
													</details>
													<details>
														<summary><b>commands</b></summary>
														<blockquote>
															<table>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/commands/check.py'>check.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/commands/show.py'>show.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/commands/hash.py'>hash.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/commands/uninstall.py'>uninstall.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/commands/debug.py'>debug.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/commands/cache.py'>cache.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/commands/completion.py'>completion.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/commands/wheel.py'>wheel.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/commands/index.py'>index.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/commands/install.py'>install.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/commands/freeze.py'>freeze.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/commands/list.py'>list.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/commands/help.py'>help.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/commands/download.py'>download.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/commands/configuration.py'>configuration.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/commands/inspect.py'>inspect.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/commands/search.py'>search.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															</table>
														</blockquote>
													</details>
													<details>
														<summary><b>locations</b></summary>
														<blockquote>
															<table>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/locations/base.py'>base.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/locations/_sysconfig.py'>_sysconfig.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															<tr>
																<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip/_internal/locations/_distutils.py'>_distutils.py</a></b></td>
																<td><code>❯ REPLACE-ME</code></td>
															</tr>
															</table>
														</blockquote>
													</details>
												</blockquote>
											</details>
										</blockquote>
									</details>
									<details>
										<summary><b>pip-24.0.dist-info</b></summary>
										<blockquote>
											<table>
											<tr>
												<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip-24.0.dist-info/LICENSE.txt'>LICENSE.txt</a></b></td>
												<td><code>❯ REPLACE-ME</code></td>
											</tr>
											<tr>
												<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip-24.0.dist-info/INSTALLER'>INSTALLER</a></b></td>
												<td><code>❯ REPLACE-ME</code></td>
											</tr>
											<tr>
												<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip-24.0.dist-info/entry_points.txt'>entry_points.txt</a></b></td>
												<td><code>❯ REPLACE-ME</code></td>
											</tr>
											<tr>
												<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip-24.0.dist-info/RECORD'>RECORD</a></b></td>
												<td><code>❯ REPLACE-ME</code></td>
											</tr>
											<tr>
												<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip-24.0.dist-info/METADATA'>METADATA</a></b></td>
												<td><code>❯ REPLACE-ME</code></td>
											</tr>
											<tr>
												<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip-24.0.dist-info/WHEEL'>WHEEL</a></b></td>
												<td><code>❯ REPLACE-ME</code></td>
											</tr>
											<tr>
												<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip-24.0.dist-info/top_level.txt'>top_level.txt</a></b></td>
												<td><code>❯ REPLACE-ME</code></td>
											</tr>
											<tr>
												<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip-24.0.dist-info/AUTHORS.txt'>AUTHORS.txt</a></b></td>
												<td><code>❯ REPLACE-ME</code></td>
											</tr>
											<tr>
												<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/training/venv/Lib/site-packages/pip-24.0.dist-info/REQUESTED'>REQUESTED</a></b></td>
												<td><code>❯ REPLACE-ME</code></td>
											</tr>
											</table>
										</blockquote>
									</details>
								</blockquote>
							</details>
						</blockquote>
					</details>
				</blockquote>
			</details>
		</blockquote>
	</details>
	<details> <!-- utils Submodule -->
		<summary><b>utils</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/utils/visualization.py'>visualization.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/utils/graph_augmentation.py'>graph_augmentation.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/sanu123-mj/gnn-graph-ai/blob/master/utils/data_loader.py'>data_loader.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			</table>
		</blockquote>
	</details>
</details>

---
##  Getting Started

###  Prerequisites

Before getting started with gnn-graph-ai, ensure your runtime environment meets the following requirements:

- **Programming Language:** Python
- **Package Manager:** Pip


###  Installation

Install gnn-graph-ai using one of the following methods:

**Build from source:**

1. Clone the gnn-graph-ai repository:
```sh
❯ git clone https://github.com/sanu123-mj/gnn-graph-ai
```

2. Navigate to the project directory:
```sh
❯ cd gnn-graph-ai
```

3. Install the project dependencies:


**Using `pip`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white" />](https://pypi.org/project/pip/)

```sh
❯ pip install -r requirements.txt
```




###  Usage
Run gnn-graph-ai using the following command:
**Using `pip`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white" />](https://pypi.org/project/pip/)

```sh
❯ python {entrypoint}
```


###  Testing
Run the test suite using the following command:
**Using `pip`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white" />](https://pypi.org/project/pip/)

```sh
❯ pytest
```


---
##  Project Roadmap

- [X] **`Task 1`**: <strike>Implement feature one.</strike>
- [ ] **`Task 2`**: Implement feature two.
- [ ] **`Task 3`**: Implement feature three.

---

##  Contributing

- **💬 [Join the Discussions](https://github.com/sanu123-mj/gnn-graph-ai/discussions)**: Share your insights, provide feedback, or ask questions.
- **🐛 [Report Issues](https://github.com/sanu123-mj/gnn-graph-ai/issues)**: Submit bugs found or log feature requests for the `gnn-graph-ai` project.
- **💡 [Submit Pull Requests](https://github.com/sanu123-mj/gnn-graph-ai/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your github account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/sanu123-mj/gnn-graph-ai
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to github**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://github.com{/sanu123-mj/gnn-graph-ai/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=sanu123-mj/gnn-graph-ai">
   </a>
</p>
</details>

---

##  License

This project is protected under the [SELECT-A-LICENSE](https://choosealicense.com/licenses) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

---

##  Acknowledgments

- List any resources, contributors, inspiration, etc. here.

---
