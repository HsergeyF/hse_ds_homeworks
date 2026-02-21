from DataProvider import DataProvider
from Visualizer import Visualizer
from Models import PlainClassifier, DimensionReduction


if __name__ == "__main__":

    data_provider = DataProvider('./data/data.parquet')
    models = PlainClassifier(data_provider)
    dimReducer = DimensionReduction(data_provider)
    visualizer = Visualizer(data_provider)
    
    models.get_decision_tree_clf()

    visualizer.plot_initial_warning_data("static/warning_features.png",is_show=True) \
              .plot_aprox_dist("static/aprox_dist.png", is_show=True)\
              .plot_common_stats("static/common_stat.png", is_show=True,is_log_valuable_pairs=False) \
              .plot_t_sne("./static/t-sne.png",dimReducer.get_TCNA()) \
              .plot_plain_decision_tree("./static/decision_tree.png", models.decision_tree)

    models.compare_models()


