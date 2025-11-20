#!/usr/bin/env python
# coding: utf-8

# In[1]:
import glob
import os
import re
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical

def load_image_dataset(
        image_dir,
        labels_file,
        max_images=None,
        target_size=(256, 256),
        num_classes=5):

    # Load CSV labels
    labels_df = pd.read_csv(labels_file)
    labels_dict = dict(zip(labels_df['image'], labels_df['level']))

    # Get all images recursively
    all_files = glob.glob(os.path.join(image_dir, "**", "*.jpeg"), recursive=True)

    images = []
    labels = []
    count = 0

    for filepath in all_files:
        if max_images is not None and count >= max_images:
            break

        filename = os.path.basename(filepath)

        # Load image
        img = load_img(filepath, target_size=target_size)
        img_array = img_to_array(img) / 255.0

        # Match label
        base_name = re.sub(r'\s*\(.*\)', '', filename.split('.')[0])
        label = labels_dict.get(base_name)

        if label is not None:
            images.append(img_array)
            labels.append(label)
            count += 1

    # Convert to arrays
    images = np.array(images)
    labels = np.array(labels)

    # One-hot encode
    labels = to_categorical(labels, num_classes=num_classes)
    print("Loaded", images.shape[0], "images with labels")

    return images, labels


# In[2]:


def prepare_data(images, labels):
    labels_flat = np.argmax(labels, axis=1)
    images_flat = images.reshape((images.shape[0], -1))
    print(f'images shape {images_flat.shape}')
    print(f'labels shape {labels_flat.shape}')
    
    return images_flat, labels_flat


# In[3]:


import numpy as np
import ray
import time

@ray.remote
class Worker:
    """Ray worker for distributed K-Means computation"""

    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def compute_distances_and_assign(self, X_chunk, centroids):
        """
        Memory-efficient distance computation using batch processing
        """
        import gc
        
        t_start = time.time()
        
        centroids = np.array(centroids)
        X_chunk = np.array(X_chunk)
        
        n_samples = X_chunk.shape[0]
        labels = np.zeros(n_samples, dtype=int)
        
        # Process in batches to avoid large intermediate arrays
        batch_size = min(1000, n_samples)
        
        t_distance_start = time.time()
        for i in range(0, n_samples, batch_size):
            batch = X_chunk[i:i+batch_size]
            
            # Compute distances for this batch only
            distances = np.sqrt(((batch[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
            labels[i:i+batch_size] = np.argmin(distances, axis=1)
            
            del distances
            gc.collect()
        
        t_distance_end = time.time()

        # Compute partial sums and counts for each cluster
        t_aggregation_start = time.time()
        partial_sums = np.zeros((self.n_clusters, X_chunk.shape[1]))
        counts = np.zeros(self.n_clusters, dtype=int)

        for k in range(self.n_clusters):
            mask = labels == k
            counts[k] = np.sum(mask)
            if counts[k] > 0:
                partial_sums[k] = X_chunk[mask].sum(axis=0)
        
        t_aggregation_end = time.time()
        t_total = t_aggregation_end - t_start
        
        print(f'Worker - Distance computation: {t_distance_end-t_distance_start:.3f}s | '
              f'Aggregation: {t_aggregation_end-t_aggregation_start:.3f}s | '
              f'Total: {t_total:.3f}s')

        return labels, partial_sums, counts



class K_Means_Distributed:
    """Distributed K-Means using Ray for parallel computation"""

    def __init__(self, n_clusters, max_iter=300, tol=1e-4, num_workers=4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.num_workers = num_workers
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.number_of_iter = 0

    def _initialize_centroids(self, X):
        """Initialize centroids to zero"""
        return np.zeros((self.n_clusters, X.shape[1]))

    def fit(self, X):
        """Train distributed K-Means on data X"""
        t_fit_start = time.time()
        
        X = np.array(X)

        # Create workers
        t_worker_start = time.time()
        workers = [Worker.remote(self.n_clusters) for _ in range(self.num_workers)]
        t_worker_end = time.time()
        #print(f'\n=== Setup ===')
        #print(f'Worker creation: {t_worker_end-t_worker_start:.3f}s')

        # Initialize centroids
        self.cluster_centers_ = self._initialize_centroids(X)

        # Split data into chunks for workers
        t_split_start = time.time()
        X_chunks = np.array_split(X, self.num_workers)
        chunk_refs = [ray.put(chunk) for chunk in X_chunks]
        t_split_end = time.time()
        print(f'Data splitting & Ray.put: {t_split_end-t_split_start:.3f}s')
        
        # Timing collectors
        collect_it = []
        dispatch_times = []
        aggregation_times = []
        centroid_update_times = []
        convergence_check_times = []

        print(f'\n=== Training Started ===')
        for iteration in range(self.max_iter):
            print(f'\n--- Iteration {iteration + 1} ---')
            
            # Distribute work: compute distances and assign clusters
            t_dispatch_start = time.time()
            futures = [
                worker.compute_distances_and_assign.remote(chunk_ref, self.cluster_centers_)
                for worker, chunk_ref in zip(workers, chunk_refs)
            ]
            t_dispatch_end = time.time()
            dispatch_times.append(t_dispatch_end - t_dispatch_start)
            
            # Wait for results (this is where parallel work happens)
            t_collect_start = time.time()
            results = ray.get(futures)
            t_collect_end = time.time()
            collect_time = t_collect_end - t_collect_start
            collect_it.append(collect_time)
            print(f'Ray.get (collection): {collect_time:.3f}s')

            # Aggregate results from all workers
            t_agg_start = time.time()
            all_labels = []
            total_sums = np.zeros((self.n_clusters, X.shape[1]))
            total_counts = np.zeros(self.n_clusters, dtype=int)

            for labels_chunk, partial_sums, counts in results:
                all_labels.extend(labels_chunk)
                total_sums += partial_sums
                total_counts += counts

            self.labels_ = np.array(all_labels)
            t_agg_end = time.time()
            aggregation_times.append(t_agg_end - t_agg_start)
            print(f'Result aggregation: {t_agg_end-t_agg_start:.3f}s')

            # Compute new centroids from aggregated data
            t_centroid_start = time.time()
            new_centers = np.zeros((self.n_clusters, X.shape[1]))
            for k in range(self.n_clusters):
                if total_counts[k] > 0:
                    new_centers[k] = total_sums[k] / total_counts[k]
                else:
                    # Handle empty cluster - reinitialize randomly
                    new_centers[k] = X[k % X.shape[0]]
            t_centroid_end = time.time()
            centroid_update_times.append(t_centroid_end - t_centroid_start)
            print(f'Centroid update: {t_centroid_end-t_centroid_start:.3f}s')

            # Check convergence
            t_conv_start = time.time()
            if self.tol is not None:
                center_shift = np.sqrt(((new_centers - self.cluster_centers_) ** 2).sum())
                self.cluster_centers_ = new_centers
                t_conv_end = time.time()
                convergence_check_times.append(t_conv_end - t_conv_start)
                print(f'Convergence check: {t_conv_end-t_conv_start:.3f}s | Center shift: {center_shift:.6f}')

                if center_shift < self.tol:
                    self.number_of_iter = iteration + 1
                    print(f'\n✓ Converged after {iteration + 1} iterations')
                    break
            else:
                self.cluster_centers_ = new_centers
                t_conv_end = time.time()
                convergence_check_times.append(t_conv_end - t_conv_start)
        else:
            self.number_of_iter = self.max_iter
            print(f'\n⚠ Max iterations ({self.max_iter}) reached')

        t_fit_end = time.time()
        
        # Print summary
        print(f'\n=== Training Summary ===')
        print(f'Total training time: {t_fit_end-t_fit_start:.3f}s')
        print(f'Number of iterations: {self.number_of_iter}')
        print(f'\nCollection times per iteration: {[f"{t:.3f}s" for t in collect_it]}')
        print(f'Average collection time: {np.mean(collect_it):.3f}s')
        print(f'Somme collection time: {np.sum(collect_it):.3f}s')
        print(f'Average dispatch time: {np.mean(dispatch_times):.3f}s')
        print(f'Average aggregation time: {np.mean(aggregation_times):.3f}s')
        print(f'Average centroid update time: {np.mean(centroid_update_times):.3f}s')
        print(f'Average convergence check time: {np.mean(convergence_check_times):.3f}s')

        return self

    def predict(self, X):
        """Predict cluster labels for new data (non-distributed for simplicity)"""
        X = np.array(X)
        distances = np.sqrt(((X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :]) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)

    def fit_predict(self, X):
        """Fit and return cluster labels"""
        self.fit(X)
        return self.labels_


# In[ ]:



