"""
Adapted from https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py
"""


import inspect
import os
import random
import tempfile
import itertools
import logging
from typing import Any, Dict, List, Iterator, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from torch_geometric.data.dataset import Dataset

logger = logging.getLogger(__name__)


def randomize_arrays(array_list):
  # assumes that every array is of the same dimension
  num_rows = array_list[0].shape[0]
  perm = np.random.permutation(num_rows)
  permuted_arrays = []
  for array in array_list:
    permuted_arrays.append(array[perm])
  return permuted_arrays


class RandomSplitter(object):
  """Class for doing random data splits.
  """

  def split(
      self,
      dataset: Dataset,
      frac_train: float = 0.7,
      frac_valid: float = 0.1,
      frac_test: float = 0.2,
      seed: Optional[int] = None,
      log_every_n: Optional[int] = None
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits internal compounds randomly into train/validation/test.

    Parameters
    ----------
    dataset: Dataset
      Dataset to be split.
    seed: int, optional (default None)
      Random seed to use.
    frac_train: float, optional (default 0.8)
      The fraction of data to be used for the training split.
    frac_valid: float, optional (default 0.1)
      The fraction of data to be used for the validation split.
    frac_test: float, optional (default 0.1)
      The fraction of data to be used for the test split.
    seed: int, optional (default None)
      Random seed to use.
    log_every_n: int, optional (default None)
      Log every n examples (not currently used).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
      A tuple of train indices, valid indices, and test indices.
      Each indices is a numpy array.
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
    if seed is not None:
      np.random.seed(seed)
    num_datapoints = len(dataset)
    train_cutoff = int(frac_train * num_datapoints)
    valid_cutoff = int((frac_train + frac_valid) * num_datapoints)
    shuffled = np.random.permutation(range(num_datapoints))
    return (shuffled[:train_cutoff], shuffled[train_cutoff:valid_cutoff],
            shuffled[valid_cutoff:])


class RandomGroupSplitter(object):
  """Random split based on groupings.

  A splitter class that splits on groupings. An example use case is when
  there are multiple conformations of the same molecule that share the same
  topology.  This splitter subsequently guarantees that resulting splits
  preserve groupings.

  Note that it doesn't do any dynamic programming or something fancy to try
  to maximize the choice such that frac_train, frac_valid, or frac_test is
  maximized.  It simply permutes the groups themselves. As such, use with
  caution if the number of elements per group varies significantly.
  """

  def __init__(self, groups: Sequence):
    """Initialize this object.

    Parameters
    ----------
    groups: Sequence
      An array indicating the group of each item.
      The length is equals to `len(dataset.X)`

    Note
    ----
    The examples of groups is the following.

    | groups    : 3 2 2 0 1 1 2 4 3
    | dataset.X : 0 1 2 3 4 5 6 7 8

    | groups    : a b b e q x a a r
    | dataset.X : 0 1 2 3 4 5 6 7 8
    """
    self.groups = groups

  def split(
      self,
      dataset: Dataset,
      frac_train: float = 0.8,
      frac_valid: float = 0.1,
      frac_test: float = 0.1,
      seed: Optional[int] = None,
      log_every_n: Optional[int] = None
  ) -> Tuple[List[int], List[int], List[int]]:
    """Return indices for specified split

    Parameters
    ----------
    dataset: Dataset
      Dataset to be split.
    frac_train: float, optional (default 0.8)
      The fraction of data to be used for the training split.
    frac_valid: float, optional (default 0.1)
      The fraction of data to be used for the validation split.
    frac_test: float, optional (default 0.1)
      The fraction of data to be used for the test split.
    seed: int, optional (default None)
      Random seed to use.
    log_every_n: int, optional (default None)
      Log every n examples (not currently used).

    Returns
    -------
    Tuple[List[int], List[int], List[int]]
      A tuple `(train_inds, valid_inds, test_inds` of the indices (integers) for
      the various splits.
    """

    assert len(self.groups) == dataset.X.shape[0]
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)

    if seed is not None:
      np.random.seed(seed)

    # dict is needed in case groups aren't strictly flattened or
    # hashed by something non-integer like
    group_dict: Dict[Any, List[int]] = {}
    for idx, g in enumerate(self.groups):
      if g not in group_dict:
        group_dict[g] = []
      group_dict[g].append(idx)

    group_idxs = np.array([g for g in group_dict.values()])

    num_groups = len(group_idxs)
    train_cutoff = int(frac_train * num_groups)
    valid_cutoff = int((frac_train + frac_valid) * num_groups)
    shuffled_group_idxs = np.random.permutation(range(num_groups))

    train_groups = shuffled_group_idxs[:train_cutoff]
    valid_groups = shuffled_group_idxs[train_cutoff:valid_cutoff]
    test_groups = shuffled_group_idxs[valid_cutoff:]

    train_idxs = list(itertools.chain(*group_idxs[train_groups]))
    valid_idxs = list(itertools.chain(*group_idxs[valid_groups]))
    test_idxs = list(itertools.chain(*group_idxs[test_groups]))

    return train_idxs, valid_idxs, test_idxs


class RandomStratifiedSplitter(object):
  """RandomStratified Splitter class.

  For sparse multitask datasets, a standard split offers no guarantees
  that the splits will have any active compounds. This class tries to
  arrange that each split has a proportional number of the actives for each
  task. This is strictly guaranteed only for single-task datasets, but for
  sparse multitask datasets it usually manages to produces a fairly accurate
  division of the actives for each task.

  Note
  ----
  This splitter is primarily designed for boolean labeled data. It considers
  only whether a label is zero or non-zero. When labels can take on multiple
  non-zero values, it does not try to give each split a proportional fraction
  of the samples with each value.
  """

  def split(self,
            dataset: Dataset,
            frac_train: float = 0.8,
            frac_valid: float = 0.1,
            frac_test: float = 0.1,
            seed: Optional[int] = None,
            log_every_n: Optional[int] = None) -> Tuple:
    """Return indices for specified split

    Parameters
    ----------
    dataset: dc.data.Dataset
      Dataset to be split.
    seed: int, optional (default None)
      Random seed to use.
    frac_train: float, optional (default 0.8)
      The fraction of data to be used for the training split.
    frac_valid: float, optional (default 0.1)
      The fraction of data to be used for the validation split.
    frac_test: float, optional (default 0.1)
      The fraction of data to be used for the test split.
    log_every_n: int, optional (default None)
      Controls the logger by dictating how often logger outputs
      will be produced.

    Returns
    -------
    Tuple
      A tuple `(train_inds, valid_inds, test_inds)` of the indices (integers) for
      the various splits.
    """
    y_present = (dataset.y != 0) * (dataset.w != 0)
    if len(y_present.shape) == 1:
      y_present = np.expand_dims(y_present, 1)
    elif len(y_present.shape) > 2:
      raise ValueError(
          'RandomStratifiedSplitter cannot be applied when y has more than two dimensions'
      )
    if seed is not None:
      np.random.seed(seed)

    # Figure out how many positive samples we want for each task in each dataset.

    n_tasks = y_present.shape[1]
    indices_for_task = [
        np.random.permutation(np.nonzero(y_present[:, i])[0])
        for i in range(n_tasks)
    ]
    count_for_task = np.array([len(x) for x in indices_for_task])
    train_target = np.round(frac_train * count_for_task).astype(int)
    valid_target = np.round(frac_valid * count_for_task).astype(int)
    test_target = np.round(frac_test * count_for_task).astype(int)

    # Assign the positive samples to datasets.  Since a sample may be positive
    # on more than one task, we need to keep track of the effect of each added
    # sample on each task.  To try to keep everything balanced, we cycle through
    # tasks, assigning one positive sample for each one.

    train_counts = np.zeros(n_tasks, int)
    valid_counts = np.zeros(n_tasks, int)
    test_counts = np.zeros(n_tasks, int)
    set_target = [train_target, valid_target, test_target]
    set_counts = [train_counts, valid_counts, test_counts]
    set_inds: List[List[int]] = [[], [], []]
    assigned = set()
    max_count = np.max(count_for_task)
    for i in range(max_count):
      for task in range(n_tasks):
        indices = indices_for_task[task]
        if i < len(indices) and indices[i] not in assigned:
          # We have a sample that hasn't been assigned yet.  Assign it to
          # whichever set currently has the lowest fraction of its target for
          # this task.

          index = indices[i]
          set_frac = [
              1 if set_target[i][task] == 0 else set_counts[i][task] /
              set_target[i][task] for i in range(3)
          ]
          set_index = np.argmin(set_frac)
          set_inds[set_index].append(index)
          assigned.add(index)
          set_counts[set_index] += y_present[index]

    # The remaining samples are negative for all tasks.  Add them to fill out
    # each set to the correct total number.

    n_samples = y_present.shape[0]
    set_size = [
        int(np.round(n_samples * f))
        for f in (frac_train, frac_valid, frac_test)
    ]
    s = 0
    for i in np.random.permutation(range(n_samples)):
      if i not in assigned:
        while s < 2 and len(set_inds[s]) >= set_size[s]:
          s += 1
        set_inds[s].append(i)
    return tuple(sorted(x) for x in set_inds)


class SingletaskStratifiedSplitter(object):
  """Class for doing data splits by stratification on a single task.
  """

  def __init__(self, task_number: int = 0):
    """
    Creates splitter object.

    Parameters
    ----------
    task_number: int, optional (default 0)
      Task number for stratification.
    """
    self.task_number = task_number

  # FIXME: Signature of "k_fold_split" incompatible with supertype "Splitter"
  def k_fold_split(  # type: ignore [override]
      self,
      dataset: Dataset,
      k: int,
      directories: Optional[List[str]] = None,
      seed: Optional[int] = None,
      log_every_n: Optional[int] = None,
      **kwargs) -> List[Dataset]:
    """
    Splits compounds into k-folds using stratified sampling.
    Overriding base class k_fold_split.

    Parameters
    ----------
    dataset: Dataset
      Dataset to be split.
    k: int
      Number of folds to split `dataset` into.
    directories: List[str], optional (default None)
      List of length k filepaths to save the result disk-datasets.
    seed: int, optional (default None)
      Random seed to use.
    log_every_n: int, optional (default None)
      Log every n examples (not currently used).

    Returns
    -------
    fold_datasets: List[Dataset]
      List of dc.data.Dataset objects
    """
    logger.info("Computing K-fold split")
    if directories is None:
      directories = [tempfile.mkdtemp() for _ in range(k)]
    else:
      assert len(directories) == k

    y_s = dataset.y[:, self.task_number]
    sortidx = np.argsort(y_s)
    sortidx_list = np.array_split(sortidx, k)

    fold_datasets = []
    for fold in range(k):
      fold_dir = directories[fold]
      fold_ind = sortidx_list[fold]
      fold_dataset = dataset.select(fold_ind, fold_dir)
      fold_datasets.append(fold_dataset)
    return fold_datasets

  def split(
      self,
      dataset: Dataset,
      frac_train: float = 0.8,
      frac_valid: float = 0.1,
      frac_test: float = 0.1,
      seed: Optional[int] = None,
      log_every_n: Optional[int] = None
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits compounds into train/validation/test using stratified sampling.

    Parameters
    ----------
    dataset: Dataset
      Dataset to be split.
    frac_train: float, optional (default 0.8)
      Fraction of dataset put into training data.
    frac_valid: float, optional (default 0.1)
      Fraction of dataset put into validation data.
    frac_test: float, optional (default 0.1)
      Fraction of dataset put into test data.
    seed: int, optional (default None)
      Random seed to use.
    log_every_n: int, optional (default None)
      Log every n examples (not currently used).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
      A tuple of train indices, valid indices, and test indices.
      Each indices is a numpy array.
    """
    # JSG Assert that split fractions can be written as proper fractions over 10.
    # This can be generalized in the future with some common demoninator determination.
    # This will work for 80/20 train/test or 80/10/10 train/valid/test (most use cases).
    np.testing.assert_equal(frac_train + frac_valid + frac_test, 1.)
    np.testing.assert_equal(10 * frac_train + 10 * frac_valid + 10 * frac_test,
                            10.)

    if seed is not None:
      np.random.seed(seed)

    y_s = dataset.y[:, self.task_number]
    sortidx = np.argsort(y_s)

    split_cd = 10
    train_cutoff = int(np.round(frac_train * split_cd))
    valid_cutoff = int(np.round(frac_valid * split_cd)) + train_cutoff

    train_idx: np.ndarray = np.array([])
    valid_idx: np.ndarray = np.array([])
    test_idx: np.ndarray = np.array([])

    while sortidx.shape[0] >= split_cd:
      sortidx_split, sortidx = np.split(sortidx, [split_cd])
      shuffled = np.random.permutation(range(split_cd))
      train_idx = np.hstack([train_idx, sortidx_split[shuffled[:train_cutoff]]])
      valid_idx = np.hstack(
          [valid_idx, sortidx_split[shuffled[train_cutoff:valid_cutoff]]])
      test_idx = np.hstack([test_idx, sortidx_split[shuffled[valid_cutoff:]]])

    # Append remaining examples to train
    if sortidx.shape[0] > 0:
      np.hstack([train_idx, sortidx])

    return (train_idx, valid_idx, test_idx)


class IndexSplitter(object):
  """Class for simple order based splits.

  Use this class when the `Dataset` you have is already ordered sa you would
  like it to be processed. Then the first `frac_train` proportion is used for
  training, the next `frac_valid` for validation, and the final `frac_test` for
  testing. This class may make sense to use your `Dataset` is already time
  ordered (for example).
  """

  def split(
      self,
      dataset: Dataset,
      frac_train: float = 0.8,
      frac_valid: float = 0.1,
      frac_test: float = 0.1,
      seed: Optional[int] = None,
      log_every_n: Optional[int] = None
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Splits internal compounds into train/validation/test in provided order.

    Parameters
    ----------
    dataset: Dataset
      Dataset to be split.
    frac_train: float, optional (default 0.8)
      The fraction of data to be used for the training split.
    frac_valid: float, optional (default 0.1)
      The fraction of data to be used for the validation split.
    frac_test: float, optional (default 0.1)
      The fraction of data to be used for the test split.
    seed: int, optional (default None)
      Random seed to use.
    log_every_n: int, optional
      Log every n examples (not currently used).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
      A tuple of train indices, valid indices, and test indices.
      Each indices is a numpy array.
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
    num_datapoints = len(dataset)
    train_cutoff = int(frac_train * num_datapoints)
    valid_cutoff = int((frac_train + frac_valid) * num_datapoints)
    indices = np.arange(num_datapoints)
    return (indices[:train_cutoff], indices[train_cutoff:valid_cutoff],
            indices[valid_cutoff:])


class SpecifiedSplitter(object):
  """Split data in the fashion specified by user.

  For some applications, you will already know how you'd like to split the
  dataset. In this splitter, you simplify specify `valid_indices` and
  `test_indices` and the datapoints at those indices are pulled out of the
  dataset. Note that this is different from `IndexSplitter` which only splits
  based on the existing dataset ordering, while this `SpecifiedSplitter` can
  split on any specified ordering.
  """

  def __init__(self,
               valid_indices: Optional[List[int]] = None,
               test_indices: Optional[List[int]] = None):
    """
    Parameters
    -----------
    valid_indices: List[int]
      List of indices of samples in the valid set
    test_indices: List[int]
      List of indices of samples in the test set
    """
    self.valid_indices = valid_indices
    self.test_indices = test_indices

  def split(
      self,
      dataset: Dataset,
      frac_train: float = 0.8,
      frac_valid: float = 0.1,
      frac_test: float = 0.1,
      seed: Optional[int] = None,
      log_every_n: Optional[int] = None
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits internal compounds into train/validation/test in designated order.

    Parameters
    ----------
    dataset: Dataset
      Dataset to be split.
    frac_train: float, optional (default 0.8)
      Fraction of dataset put into training data.
    frac_valid: float, optional (default 0.1)
      Fraction of dataset put into validation data.
    frac_test: float, optional (default 0.1)
      Fraction of dataset put into test data.
    seed: int, optional (default None)
      Random seed to use.
    log_every_n: int, optional (default None)
      Log every n examples (not currently used).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
      A tuple of train indices, valid indices, and test indices.
      Each indices is a numpy array.
    """
    num_datapoints = len(dataset)
    indices = np.arange(num_datapoints).tolist()
    train_indices = []
    if self.valid_indices is None:
      self.valid_indices = []
    if self.test_indices is None:
      self.test_indices = []
    valid_test = list(self.valid_indices)
    valid_test.extend(self.test_indices)
    for indice in indices:
      if indice not in valid_test:
        train_indices.append(indice)

    return (np.array(train_indices), np.array(self.valid_indices),
            np.array(self.test_indices))


#################################################################
# Splitter for molecule datasets
#################################################################


class MolecularWeightSplitter(object):
  """
  Class for doing data splits by molecular weight.

  Note
  ----
  This class requires RDKit to be installed.
  """

  def split(
      self,
      dataset: Dataset,
      frac_train: float = 0.8,
      frac_valid: float = 0.1,
      frac_test: float = 0.1,
      seed: Optional[int] = None,
      log_every_n: Optional[int] = None
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Splits on molecular weight.

    Splits internal compounds into train/validation/test using the MW
    calculated by SMILES string.

    Parameters
    ----------
    dataset: Dataset
      Dataset to be split.
    frac_train: float, optional (default 0.8)
      The fraction of data to be used for the training split.
    frac_valid: float, optional (default 0.1)
      The fraction of data to be used for the validation split.
    frac_test: float, optional (default 0.1)
      The fraction of data to be used for the test split.
    seed: int, optional (default None)
      Random seed to use.
    log_every_n: int, optional (default None)
      Log every n examples (not currently used).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
      A tuple of train indices, valid indices, and test indices.
      Each indices is a numpy array.
    """
    try:
      from rdkit import Chem
    except ModuleNotFoundError:
      raise ImportError("This function requires RDKit to be installed.")

    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
    if seed is not None:
      np.random.seed(seed)

    mws = []
    for smiles in dataset.ids:
      mol = Chem.MolFromSmiles(smiles)
      mw = Chem.rdMolDescriptors.CalcExactMolWt(mol)
      mws.append(mw)

    # Sort by increasing MW
    sortidx = np.argsort(mws)

    train_cutoff = int(frac_train * len(sortidx))
    valid_cutoff = int((frac_train + frac_valid) * len(sortidx))

    return (sortidx[:train_cutoff], sortidx[train_cutoff:valid_cutoff],
            sortidx[valid_cutoff:])


class MaxMinSplitter(object):
  """Chemical diversity splitter.

  Class for doing splits based on the MaxMin diversity algorithm. Intuitively,
  the test set is comprised of the most diverse compounds of the entire dataset.
  Furthermore, the validation set is comprised of diverse compounds under
  the test set.

  Note
  ----
  This class requires RDKit to be installed.
  """

  def split(
      self,
      dataset: Dataset,
      frac_train: float = 0.8,
      frac_valid: float = 0.1,
      frac_test: float = 0.1,
      seed: Optional[int] = None,
      log_every_n: Optional[int] = None
  ) -> Tuple[List[int], List[int], List[int]]:
    """
    Splits internal compounds into train/validation/test using the MaxMin diversity algorithm.

    Parameters
    ----------
    dataset: Dataset
      Dataset to be split.
    frac_train: float, optional (default 0.8)
      The fraction of data to be used for the training split.
    frac_valid: float, optional (default 0.1)
      The fraction of data to be used for the validation split.
    frac_test: float, optional (default 0.1)
      The fraction of data to be used for the test split.
    seed: int, optional (default None)
      Random seed to use.
    log_every_n: int, optional (default None)
      Log every n examples (not currently used).

    Returns
    -------
    Tuple[List[int], List[int], List[int]]
      A tuple of train indices, valid indices, and test indices.
      Each indices is a list of integers.
    """
    try:
      from rdkit import Chem, DataStructs
      from rdkit.Chem import AllChem
      from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
    except ModuleNotFoundError:
      raise ImportError("This function requires RDKit to be installed.")

    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
    if seed is None:
      seed = random.randint(0, 2**30)
    np.random.seed(seed)

    num_datapoints = len(dataset)

    train_cutoff = int(frac_train * num_datapoints)
    valid_cutoff = int((frac_train + frac_valid) * num_datapoints)

    num_valid = valid_cutoff - train_cutoff
    num_test = num_datapoints - valid_cutoff

    all_mols = []
    for ind, smiles in enumerate(dataset.ids):
      all_mols.append(Chem.MolFromSmiles(smiles))

    fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in all_mols]

    def distance(i, j):
      return 1 - DataStructs.DiceSimilarity(fps[i], fps[j])

    picker = MaxMinPicker()
    testIndices = picker.LazyPick(distFunc=distance,
                                  poolSize=num_datapoints,
                                  pickSize=num_test,
                                  seed=seed)

    validTestIndices = picker.LazyPick(distFunc=distance,
                                       poolSize=num_datapoints,
                                       pickSize=num_valid + num_test,
                                       firstPicks=testIndices,
                                       seed=seed)

    allSet = set(range(num_datapoints))
    testSet = set(testIndices)
    validSet = set(validTestIndices) - testSet

    trainSet = allSet - testSet - validSet

    assert len(testSet & validSet) == 0
    assert len(testSet & trainSet) == 0
    assert len(validSet & trainSet) == 0
    assert (validSet | trainSet | testSet) == allSet

    return sorted(list(trainSet)), sorted(list(validSet)), sorted(list(testSet))


class ButinaSplitter(object):
  """Class for doing data splits based on the butina clustering of a bulk tanimoto
  fingerprint matrix.

  Note
  ----
  This class requires RDKit to be installed.
  """

  def __init__(self, cutoff: float = 0.6):
    """Create a ButinaSplitter.

    Parameters
    ----------
    cutoff: float (default 0.6)
      The cutoff value for tanimoto similarity.  Molecules that are more similar
      than this will tend to be put in the same dataset.
    """
    super(ButinaSplitter, self).__init__()
    self.cutoff = cutoff

  def split(
      self,
      dataset: Dataset,
      frac_train: float = 0.8,
      frac_valid: float = 0.1,
      frac_test: float = 0.1,
      seed: Optional[int] = None,
      log_every_n: Optional[int] = None
  ) -> Tuple[List[int], List[int], List[int]]:
    """
    Splits internal compounds into train and validation based on the butina
    clustering algorithm. This splitting algorithm has an O(N^2) run time, where N
    is the number of elements in the dataset. The dataset is expected to be a classification
    dataset.

    This algorithm is designed to generate validation data that are novel chemotypes.
    Setting a small cutoff value will generate smaller, finer clusters of high similarity,
    whereas setting a large cutoff value will generate larger, coarser clusters of low similarity.

    Parameters
    ----------
    dataset: Dataset
      Dataset to be split.
    frac_train: float, optional (default 0.8)
      The fraction of data to be used for the training split.
    frac_valid: float, optional (default 0.1)
      The fraction of data to be used for the validation split.
    frac_test: float, optional (default 0.1)
      The fraction of data to be used for the test split.
    seed: int, optional (default None)
      Random seed to use.
    log_every_n: int, optional (default None)
      Log every n examples (not currently used).

    Returns
    -------
    Tuple[List[int], List[int], List[int]]
      A tuple of train indices, valid indices, and test indices.
    """
    try:
      from rdkit import Chem, DataStructs
      from rdkit.Chem import AllChem
      from rdkit.ML.Cluster import Butina
    except ModuleNotFoundError:
      raise ImportError("This function requires RDKit to be installed.")

    logger.info("Performing butina clustering with cutoff of", self.cutoff)
    mols = []
    for ind, smiles in enumerate(dataset.ids):
      mols.append(Chem.MolFromSmiles(smiles))
    fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in mols]

    # calcaulate scaffold sets
    # (ytz): this is directly copypasta'd from Greg Landrum's clustering example.
    dists = []
    nfps = len(fps)
    for i in range(1, nfps):
      sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
      dists.extend([1 - x for x in sims])
    scaffold_sets = Butina.ClusterData(dists,
                                       nfps,
                                       self.cutoff,
                                       isDistData=True)
    scaffold_sets = sorted(scaffold_sets, key=lambda x: -len(x))

    train_cutoff = frac_train * len(dataset)
    valid_cutoff = (frac_train + frac_valid) * len(dataset)
    train_inds: List[int] = []
    valid_inds: List[int] = []
    test_inds: List[int] = []

    logger.info("About to sort in scaffold sets")
    for scaffold_set in scaffold_sets:
      if len(train_inds) + len(scaffold_set) > train_cutoff:
        if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
          test_inds += scaffold_set
        else:
          valid_inds += scaffold_set
      else:
        train_inds += scaffold_set
    return train_inds, valid_inds, test_inds


def _generate_scaffold(smiles: str, include_chirality: bool = False) -> str:
  """Compute the Bemis-Murcko scaffold for a SMILES string.

  Bemis-Murcko scaffolds are described in DOI: 10.1021/jm9602928.
  They are essentially that part of the molecule consisting of
  rings and the linker atoms between them.

  Paramters
  ---------
  smiles: str
    SMILES
  include_chirality: bool, default False
    Whether to include chirality in scaffolds or not.

  Returns
  -------
  str
    The MurckScaffold SMILES from the original SMILES

  References
  ----------
  .. [1] Bemis, Guy W., and Mark A. Murcko. "The properties of known drugs.
     1. Molecular frameworks." Journal of medicinal chemistry 39.15 (1996): 2887-2893.

  Note
  ----
  This function requires RDKit to be installed.
  """
  try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
  except ModuleNotFoundError:
    raise ImportError("This function requires RDKit to be installed.")

  mol = Chem.MolFromSmiles(smiles)
  scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
  return scaffold


class FingerprintSplitter(object):
  """Class for doing data splits based on the Tanimoto similarity between ECFP4
  fingerprints.

  This class tries to split the data such that the molecules in each dataset are
  as different as possible from the ones in the other datasets.  This makes it a
  very stringent test of models.  Predicting the test and validation sets may
  require extrapolating far outside the training data.

  The running time for this splitter scales as O(n^2) in the number of samples.
  Splitting large datasets can take a long time.

  Note
  ----
  This class requires RDKit to be installed.
  """

  def __init__(self):
    """Create a FingerprintSplitter."""
    super(FingerprintSplitter, self).__init__()

  def split(
      self,
      dataset: Dataset,
      frac_train: float = 0.8,
      frac_valid: float = 0.1,
      frac_test: float = 0.1,
      seed: Optional[int] = None,
      log_every_n: Optional[int] = None
  ) -> Tuple[List[int], List[int], List[int]]:
    """
    Splits compounds into training, validation, and test sets based on the
    Tanimoto similarity of their ECFP4 fingerprints. This splitting algorithm
    has an O(N^2) run time, where N is the number of elements in the dataset.

    Parameters
    ----------
    dataset: Dataset
      Dataset to be split.
    frac_train: float, optional (default 0.8)
      The fraction of data to be used for the training split.
    frac_valid: float, optional (default 0.1)
      The fraction of data to be used for the validation split.
    frac_test: float, optional (default 0.1)
      The fraction of data to be used for the test split.
    seed: int, optional (default None)
      Random seed to use (ignored since this algorithm is deterministic).
    log_every_n: int, optional (default None)
      Log every n examples (not currently used).

    Returns
    -------
    Tuple[List[int], List[int], List[int]]
      A tuple of train indices, valid indices, and test indices.
    """
    try:
      from rdkit import Chem
      from rdkit.Chem import AllChem
    except ModuleNotFoundError:
      raise ImportError("This function requires RDKit to be installed.")

    # Compute fingerprints for all molecules.

    mols = [Chem.MolFromSmiles(smiles) for smiles in dataset.ids]
    fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in mols]

    # Split into two groups: training set and everything else.

    train_size = int(frac_train * len(dataset))
    valid_size = int(frac_valid * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    train_inds, test_valid_inds = _split_fingerprints(fps, train_size,
                                                      valid_size + test_size)

    # Split the second group into validation and test sets.

    if valid_size == 0 or frac_valid == 0:
      valid_inds = []
      test_inds = test_valid_inds
    elif test_size == 0 or frac_test == 0:
      test_inds = []
      valid_inds = test_valid_inds
    else:
      test_valid_fps = [fps[i] for i in test_valid_inds]
      test_inds, valid_inds = _split_fingerprints(test_valid_fps, test_size,
                                                  valid_size)
      test_inds = [test_valid_inds[i] for i in test_inds]
      valid_inds = [test_valid_inds[i] for i in valid_inds]
    return train_inds, valid_inds, test_inds


def _split_fingerprints(fps: List, size1: int,
                        size2: int) -> Tuple[List[int], List[int]]:
  """This is called by FingerprintSplitter to divide a list of fingerprints into
  two groups.
  """
  assert len(fps) == size1 + size2
  from rdkit import DataStructs

  # Begin by assigning the first molecule to the first group.

  fp_in_group = [[fps[0]], []]
  indices_in_group: Tuple[List[int], List[int]] = ([0], [])
  remaining_fp = fps[1:]
  remaining_indices = list(range(1, len(fps)))
  max_similarity_to_group = [
      DataStructs.BulkTanimotoSimilarity(fps[0], remaining_fp),
      [0] * len(remaining_fp)
  ]
  # Return identity if no tuple to split to
  if size2 == 0:
    return ((list(range(len(fps)))), [])

  while len(remaining_fp) > 0:
    # Decide which group to assign a molecule to.

    group = 0 if len(fp_in_group[0]) / size1 <= len(
        fp_in_group[1]) / size2 else 1

    # Identify the unassigned molecule that is least similar to everything in
    # the other group.

    i = np.argmin(max_similarity_to_group[1 - group])

    # Add it to the group.

    fp = remaining_fp[i]
    fp_in_group[group].append(fp)
    indices_in_group[group].append(remaining_indices[i])

    # Update the data on unassigned molecules.

    similarity = DataStructs.BulkTanimotoSimilarity(fp, remaining_fp)
    max_similarity_to_group[group] = np.delete(
        np.maximum(similarity, max_similarity_to_group[group]), i)
    max_similarity_to_group[1 - group] = np.delete(
        max_similarity_to_group[1 - group], i)
    del remaining_fp[i]
    del remaining_indices[i]
  return indices_in_group


class ScaffoldSplitter(object):
  """Class for doing data splits based on the scaffold of small molecules.

  Group  molecules  based on  the Bemis-Murcko scaffold representation, which identifies rings,
  linkers, frameworks (combinations between linkers and rings) and atomic properties  such as
  atom type, hibridization and bond order in a dataset of molecules. Then split the groups by
  the number of molecules in each group in decreasing order.

  It is necessary to add the smiles representation in the ids field during the
  Dataset creation. (Dataset.ids == smiles)

  References
  ----------
  .. [1] Bemis, Guy W., and Mark A. Murcko. "The properties of known drugs.
     1. Molecular frameworks." Journal of medicinal chemistry 39.15 (1996): 2887-2893.

  Note
  ----
  This class requires RDKit to be installed.
  """

  def split(
      self,
      dataset: Dataset,
      frac_train: float = 0.8,
      frac_valid: float = 0.1,
      frac_test: float = 0.1,
      seed: Optional[int] = None,
      log_every_n: Optional[int] = 1000
  ) -> Tuple[List[int], List[int], List[int]]:
    """
    Splits internal compounds into train/validation/test by scaffold.

    Parameters
    ----------
    dataset: Dataset
      Dataset to be split.
    frac_train: float, optional (default 0.8)
      The fraction of data to be used for the training split.
    frac_valid: float, optional (default 0.1)
      The fraction of data to be used for the validation split.
    frac_test: float, optional (default 0.1)
      The fraction of data to be used for the test split.
    seed: int, optional (default None)
      Random seed to use.
    log_every_n: int, optional (default 1000)
      Controls the logger by dictating how often logger outputs
      will be produced.

    Returns
    -------
    Tuple[List[int], List[int], List[int]]
      A tuple of train indices, valid indices, and test indices.
      Each indices is a list of integers.
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
    scaffold_sets = self.generate_scaffolds(dataset)

    train_cutoff = frac_train * len(dataset)
    valid_cutoff = (frac_train + frac_valid) * len(dataset)
    train_inds: List[int] = []
    valid_inds: List[int] = []
    test_inds: List[int] = []

    logger.info("About to sort in scaffold sets")
    for scaffold_set in scaffold_sets:
      if len(train_inds) + len(scaffold_set) > train_cutoff:
        if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
          test_inds += scaffold_set
        else:
          valid_inds += scaffold_set
      else:
        train_inds += scaffold_set
    return train_inds, valid_inds, test_inds

  def generate_scaffolds(self,
                         dataset: Dataset,
                         log_every_n: int = 1000) -> List[List[int]]:
    """Returns all scaffolds from the dataset.

    Parameters
    ----------
    dataset: Dataset
      Dataset to be split.
    log_every_n: int, optional (default 1000)
      Controls the logger by dictating how often logger outputs
      will be produced.

    Returns
    -------
    scaffold_sets: List[List[int]]
      List of indices of each scaffold in the dataset.
    """
    scaffolds = {}
    data_len = len(dataset)

    logger.info("About to generate scaffolds")
    for ind, smiles in enumerate(dataset.ids):
      if ind % log_every_n == 0:
        logger.info("Generating scaffold %d/%d" % (ind, data_len))
      scaffold = _generate_scaffold(smiles)
      if scaffold not in scaffolds:
        scaffolds[scaffold] = [ind]
      else:
        scaffolds[scaffold].append(ind)

    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    return scaffold_sets
