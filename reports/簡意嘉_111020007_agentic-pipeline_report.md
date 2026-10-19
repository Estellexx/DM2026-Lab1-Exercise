# Agentic Data Mining Pipeline Report

### Data Loading and Preprocessing
The pipeline began by loading the newsgroup dataset, which contains 2,257 documents across four categories (**load_dataset_1**). To ensure data hygiene before modeling, we checked row samples (**inspect_data_2**), verified no missing text (**check_missing_3**), and removed duplicates (**check_duplicates_4**) to prevent biased pattern mining. After checking document length statistics (**describe_data_5**), we did a sample of n=1,000 and random_status=1 (**sample_data_6**) to establish a manageable working subset.
[Document Text Length by category_name](../plots/簡意嘉_agentic-pipeline_plot_describe_data_5.png)

Next, we built a global Document-Term Matrix (**build_dtm_7**) with 2,257 documents, 35,788 terms, and 99.547% sparsity. Converting raw text into a DTM is a crucial data mining step that translates unstructured text into a structured numerical feature space. We checked term frequencies (**term_frequency_8**) and used a heatmap (**dtm_heatmap_9**) to visualize document-term distributions and sparsity.
[DTM Row Counts Heatmap](../plots/簡意嘉_agentic-pipeline_plot_dtm_heatmap_9.png)

To reduce noise and remove near-constant terms that carry little discriminative power, we applied variance filtering (**variance_filter_12**, keeping 1,710 terms at a threshold of 0.05). To further enhance feature relevance, we used Pearson (**pearson_filter_13 to 16**) and Spearman (**spearman_filter_17 to 20**) correlation filters to select features most strongly associated with each target category. Finally, a feature correlation matrix (**feature_correlation_matrix_21**) was analyzed to inspect multicollinearity among surviving terms.
[variance filter with threshold=0.01](../plots/簡意嘉_agentic-pipeline_plot_turn114.png)
[pearson filter for soc.religion.christian](../plots/簡意嘉_agentic-pipeline_plot_turn130.png)
[feature_correlation_matrix](../plots/簡意嘉_agentic-pipeline_plot_feature_correlation_matrix_21.png)

    
### Frequent Pattern Mining Across Categories
Building upon our filtered features, we conducted frequent pattern mining using the FP-Growth algorithm to discover underlying term associations and frequent itemsets within specific semantic domains. Variance-filtered mining produced extensive pattern counts (**mine_patterns_23 to 26**). To compare how different feature selection methods affect pattern discovery, we also tested TF-IDF filtering (**mine_patterns_27 to 30**) and term-frequency filtering (**mine_patterns_31 to 34**). 

### Dimensionality Reduction and Spatial Visualization
While feature selection and pattern mining handle discrete words, high-dimensional spaces suffer from the curse of dimensionality. To evaluate how effectively our features separate semantic categories in continuous space, we applied dimensionality reduction techniques. PCA provided a global linear projection (**reduce_dimensions_35** in 2D, and **reduce_dimensions_42** in 3D with 73.87% cumulative variance). To capture non-linear local neighborhood structures, we tested t-SNE with varying perplexity values (**reduce_dimensions_36** at 30, **37** at 15, and **38** at 10) alongside UMAP (**reduce_dimensions_39** and **reduce_dimensions_44**). These spatial projections successfully revealed natural cluster separations among the four categories.
[PCA](../plots/簡意嘉_agentic-pipeline_plot_reduce_dimensions_35.png)
[TSNE](../plots/簡意嘉_agentic-pipeline_plot_reduce_dimensions_36.png)
[UMPA](../plots/簡意嘉_agentic-pipeline_plot_reduce_dimensions_39.png)

### Label Binarization and Document Similarity
In the final transition from exploratory analysis to predictive modeling readiness, target category labels were transformed into asymmetric binary attributes via One-Hot Encoding (**binarize_labels_40**) to satisfy algorithm requirements. Finally, to evaluate instance-level semantic proximity within the vector space, we computed the cosine similarity between Document 0 and Document 1 (**cosine_similarity_41**). The resulting moderate score reflects expected baseline term overlap driven by common English stop words in raw text representations, successfully completing our end-to-end data mining pipeline workflow.
