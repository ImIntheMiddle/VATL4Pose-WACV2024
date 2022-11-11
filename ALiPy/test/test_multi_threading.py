from sklearn import linear_model
from sklearn.datasets import load_iris

from alipy.data_manipulate.al_split import split
from alipy.experiment import ExperimentAnalyser, State
from alipy.index.index_collections import IndexCollection
from alipy.query_strategy import QueryInstanceUncertainty
from alipy.utils.multi_thread import aceThreading

X, y = load_iris(return_X_y=True)
Train_idx, Test_idx, L_pool, U_pool = split(X=X, y=y, test_ratio=0.3, initial_label_rate=0.2, split_count=10)
ea = ExperimentAnalyser()
reg = linear_model.LogisticRegression(solver='liblinear')
qs = QueryInstanceUncertainty(X, y)


# Estimator, performanceMeasure,
def run_thread(round, train_id, test_id, Lcollection, Ucollection, saver, examples, labels, global_parameters):
    # initialize object
    reg.fit(X=examples[Lcollection.index, :], y=labels[Lcollection.index])
    pred = reg.predict(examples[test_id, :])
    accuracy = sum(pred == labels[test_id]) / len(test_id)
    # initialize StateIO module
    saver.set_initial_point(accuracy)
    while len(Ucollection) > 30:
        select_index = qs.select(Lcollection, Ucollection, model=reg)
        Ucollection.difference_update(select_index)
        Lcollection.update(select_index)

        # update model
        reg.fit(X=examples[Lcollection.index, :], y=labels[Lcollection.index])
        pred = reg.predict(examples[test_id, :])
        accuracy = sum(pred == labels[test_id]) / len(test_id)

        # save intermediate results
        st = State(select_index=select_index, performance=accuracy)
        # add user defined information
        # st.add_element(key='sub_ind', value=sub_ind)
        saver.add_state(st)
        saver.save()


mt = aceThreading(X, y, Train_idx, Test_idx,
                  [IndexCollection(i) for i in L_pool], [IndexCollection(i) for i in U_pool],
                  max_thread=None,
                  target_func=run_thread)
mt.start_all_threads()
ea.add_method(method_name='unc', method_results=mt.get_results())

print(ea)
ea.plot_learning_curves(std_area=True, show=False)
