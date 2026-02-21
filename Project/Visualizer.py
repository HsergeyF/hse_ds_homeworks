import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.tree import export_graphviz
from six import StringIO
import pydotplus
import numpy as np
import plotly.express as px

class Visualizer():
    themes = {'white': {"gridcolor": 'black',
                    "plot_bgcolor": 'white',
                    "portfolio": 'black',
                    "marker_color": 'black',
                    "table_line_color": "black",
                    "table_fill_color": "white",
                    "table_font_color": 'black'},

              'black': {"gridcolor": 'white',
                    "plot_bgcolor": 'black',
                    "portfolio": 'white',
                    "marker_color": 'white',
                    "table_line_color": "white",
                    "table_fill_color": 'black',
                    "table_font_color": 'white'}}
    
    _warning_grid_spec = [
        [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
        [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]
    ]

    _approx_grid_spec = [
        [{"type": "xy"},{"type": "xy", 'rowspan': 2}],
        [{"type": "xy"}, None]
    ]

    _common_stats_spec = [
        [{"type": "xy", 'rowspan': 2}, {"type": "pie"}],
        [None,{"type": "xy"}] 
    ]

    def __init__(self, data_provider, theme='white'):
        self.data_provider = data_provider
        self.theme= theme
    
    def configure_layout(self, figure, title, is_upd_trace=True):
        if is_upd_trace:
            figure.update_traces(marker_color=self.themes[self.theme]["plot_bgcolor"],
                                marker_line_color=self.themes[self.theme]["marker_color"],
                                marker_line_width=2)
        figure.update_layout(height=1400, width=1600)
        figure.update_layout(
                title_text=title,
                title_x=0.5,
                plot_bgcolor='rgba(0,0,0,0)',
                font_family="Inter",
                font_size=12,
                font_color="#090F24")
        return figure
    
    def plot_initial_warning_data(self, path, is_show = False):
        figure = make_subplots(rows=2,cols=3, specs=self._warning_grid_spec,
                                subplot_titles=self.data_provider.warning_features
                            )
        r,c = (1,1)
        for feat in self.data_provider.warning_features:
            trace = go.Bar(x=self.data_provider.X.index, y=self.data_provider.X[feat], 
                           marker_color='black', showlegend=False)
            figure.add_trace(trace, row=r, col=c)
            c += 1
            if c == 4: 
                c = 1
                r += 1
    
        figure = self.configure_layout(figure, "Warning features values")
        figure.write_image(path)
        if is_show: figure.show()
        return self
    
    def plot_aprox_dist(self,path, is_show = False):
        non_normal_feature, dist_type = self.data_provider.get_random_non_norm_features(1)[0]
        figure = make_subplots(rows=2,cols=2, specs=self._approx_grid_spec,
                                subplot_titles=[f'Original distribution {non_normal_feature}, {dist_type}', 
                                                'Features dist count',
                                                f'Log distribution {non_normal_feature}']
                               )
     
        distr_count = self.data_provider.distributions.groupby('distr').count()

        trace = go.Histogram(x=self.data_provider.X[non_normal_feature], 
                        showlegend=False)
        figure.add_trace(trace, row=1, col=1)

        trace = go.Histogram(x=self.data_provider.X[non_normal_feature].apply(lambda x: np.log(x)),
                        showlegend=False)
        figure.add_trace(trace, row=2, col=1)

        trace = go.Bar(x=distr_count.index, y=distr_count['feature'],  
                        showlegend=False)
        figure.add_trace(trace, row=1, col=2)
        figure = self.configure_layout(figure, "Features distribution analysis")
        figure.write_image(path)
        if is_show: figure.show()
        return self
    
    def plot_common_stats(self,path, is_show=False, is_log_valuable_pairs= False):
        figure = make_subplots(rows=2,cols=2, 
                               specs=self._common_stats_spec,
                               subplot_titles=['Kendall correlation matrix',
                                               'Percent of values in every targer class',
                                               'Average target value in every comlexity class' ]
                               )
        trace = go.Pie(labels=['Positive', 'Negative'],
                       values=[self.data_provider.get_target_class_count(1), 
                               self.data_provider.get_target_class_count(0)],
                       showlegend=False, marker_colors=['black', 'white'])
        figure.add_trace(trace, row=1, col=2)
        
        trace = go.Bar(x=self.data_provider.complexity_distr.index,
                        y=self.data_provider.complexity_distr.values,
                       marker_color=self.themes[self.theme]["plot_bgcolor"],
                       marker_line_color=self.themes[self.theme]["marker_color"],
                       marker_line_width=2, showlegend=False)
        figure.add_trace(trace, row=2, col=2)
        corr = self.data_provider.X.corr(method="kendall")
        heat = go.Heatmap(
                    z = corr,
                    x = corr.columns.values,
                    y = corr.columns.values,
                    zmin = - 0.25, 
                    zmax = 1,
                    xgap = 1, 
                    ygap = 1,
                    colorscale='gray'
        )
        figure.add_trace(heat,  row=1, col=1)
        figure = self.configure_layout(figure, "Common statistics", is_upd_trace=False)
        figure.write_image(path)
        if is_show:  figure.show()
        if is_log_valuable_pairs:
            valuable_pairs = corr.abs().unstack()
            valuable_pairs = valuable_pairs.sort_values(kind="quicksort")
            valuable_pairs = valuable_pairs[valuable_pairs.values>.95]
            valuable_pairs = valuable_pairs[valuable_pairs.values<1]
        return self
    
    def plot_plain_decision_tree(self, path, model):
        dot_data = StringIO()
        export_graphviz(model, out_file=dot_data,  
                        filled=False, rounded=True,
                        special_characters=True,
                        feature_names = self.data_provider.features,
                        class_names=['0','1'])
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
        graph.write_png(path)
        return self

    def plot_t_sne(self,path,  X_pca):
       
        figure = px.scatter(x=X_pca[0], y=X_pca[1], 
                            color=self.data_provider.y_train,
                            color_continuous_scale='greys')
        
        figure.update_layout(
            xaxis_title="First Component",
            yaxis_title="Second Component",
        )
        figure.update_traces(marker=dict(line=dict(width=2,
                                         color='DarkSlateGrey')
                                         ),
                            selector=dict(mode='markers'))
        figure= self.configure_layout(figure, "T-sne visualization", is_upd_trace=False)
        figure.write_image(path)
        figure.show()
        return self




