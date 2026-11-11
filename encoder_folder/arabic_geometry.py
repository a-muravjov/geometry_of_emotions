"""Multilingual emotion geometry experiments for Arabic for the 3 encoder models.
"""

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModel
    from tqdm import tqdm
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, f1_score
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    DATA_FILE = "combined_arabic_balanced_1200_new_bestest_no_@_1041_laughing.csv"
    MAX_LENGTH = 128
    BATCH_SIZE = 32
    RANDOM_STATE = 42

    # Pooling

    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        summed = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        counts = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return summed / counts

    def cls_pooling(model_output):
        return model_output.last_hidden_state[:, 0, :]

    # Encoding
    def encode_texts(text_list, tokenizer, model, device, batch_size=32, max_length=128):
        mean_embs = []
        cls_embs = []

        with torch.no_grad():
            for i in tqdm(range(0, len(text_list), batch_size), desc="Encoding"):
                batch_texts = text_list[i:i + batch_size]

                encoded = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}

                outputs = model(**encoded)

                mean_emb = mean_pooling(outputs, encoded["attention_mask"])
                cls_emb = cls_pooling(outputs)

                mean_embs.append(mean_emb.cpu().numpy())
                cls_embs.append(cls_emb.cpu().numpy())

        return np.vstack(mean_embs), np.vstack(cls_embs)

    # Centroids

    def compute_centroids(embeddings, df, emotion_order):
        centroids = []
        for emotion in emotion_order:
            mask = (df["emotion"] == emotion).values
            centroids.append(embeddings[mask].mean(axis=0))
        centroids = np.vstack(centroids)
        return normalize(centroids, norm="l2")

    # PCA

    def plot_centroids_in_full_pca_2d(embeddings, centroids, emotion_order, title):
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        pca.fit(embeddings)
        centroids_2d = pca.transform(centroids)

        print(f"\n{title}")
        print("Explained variance ratio:", pca.explained_variance_ratio_.round(4))

        plt.figure(figsize=(8, 6))
        for i, emotion in enumerate(emotion_order):
            plt.scatter(centroids_2d[i, 0], centroids_2d[i, 1], s=160)
            plt.text(
                centroids_2d[i, 0],
                centroids_2d[i, 1],
                emotion,
                fontsize=11,
                ha='center',
                va='center'
            )

        plt.title(title)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(alpha=0.25)
        plt.show()

    def plot_centroids_in_full_pca_3d(embeddings, centroids, emotion_order, title):
        pca = PCA(n_components=3, random_state=RANDOM_STATE)
        pca.fit(embeddings)
        centroids_3d = pca.transform(centroids)

        print(f"\n{title}")
        print("Explained variance ratio:", pca.explained_variance_ratio_.round(4))

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        for i, emotion in enumerate(emotion_order):
            ax.scatter(
                centroids_3d[i, 0],
                centroids_3d[i, 1],
                centroids_3d[i, 2],
                s=140
            )
            ax.text(
                centroids_3d[i, 0],
                centroids_3d[i, 1],
                centroids_3d[i, 2],
                emotion,
                fontsize=11
            )

        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        plt.show()

    # Probe training
    def train_probe(embeddings, labels, name):
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings,
            labels,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=labels
        )

        clf = LogisticRegression(
            max_iter=3000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
            class_weight="balanced"
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        print(f"\n{name} probe")
        print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
        print("Balanced accuracy:", round(balanced_accuracy_score(y_test, y_pred), 4))
        print("Macro F1:", round(f1_score(y_test, y_pred, average="macro"), 4))
        print("Weighted F1:", round(f1_score(y_test, y_pred, average="weighted"), 4))
        print(classification_report(y_test, y_pred, digits=3))

        return clf

    # Linear probe PCA
    def get_probe_space(probe, embeddings, use_probabilities=False):
        if use_probabilities:
            return probe.predict_proba(embeddings)
        else:
            scores = probe.decision_function(embeddings)

            if scores.ndim == 1:
                scores = scores[:, None]

            return scores


    def plot_probe_pca_2d(probe, embeddings, labels, emotion_order, title, use_probabilities=False):
        probe_space = get_probe_space(probe, embeddings, use_probabilities=use_probabilities)

        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        X_2d = pca.fit_transform(probe_space)

        print(f"\n{title}")
        print("Probe space shape:", probe_space.shape)
        print("Explained variance ratio:", pca.explained_variance_ratio_.round(4))

        plt.figure(figsize=(9, 7))
        for emotion in emotion_order:
            mask = (np.array(labels) == emotion)
            plt.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                label=emotion,
                alpha=0.35,
                s=10
            )

        plt.title(title)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
        plt.grid(alpha=0.2)
        plt.show()


    def plot_probe_centroids_pca_2d(probe, embeddings, df, emotion_order, title, use_probabilities=False):
        probe_space = get_probe_space(probe, embeddings, use_probabilities=use_probabilities)

        # centroid in probe space
        centroids = []
        for emotion in emotion_order:
            mask = (df["emotion"] == emotion).values
            centroids.append(probe_space[mask].mean(axis=0))
        centroids = np.vstack(centroids)

        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        pca.fit(probe_space)
        centroids_2d = pca.transform(centroids)

        print(f"\n{title}")
        print("Probe space shape:", probe_space.shape)
        print("Explained variance ratio:", pca.explained_variance_ratio_.round(4))

        dx = (centroids_2d[:, 0].max() - centroids_2d[:, 0].min()) * 0.02
        dy = (centroids_2d[:, 1].max() - centroids_2d[:, 1].min()) * 0.02

        plt.figure(figsize=(8, 6))
        for i, emotion in enumerate(emotion_order):
            plt.scatter(centroids_2d[i, 0], centroids_2d[i, 1], s=180)
            plt.text(
                centroids_2d[i, 0] + dx,
                centroids_2d[i, 1] + dy,
                emotion,
                fontsize=11
            )

        plt.title(title)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(alpha=0.25)
        plt.show()


    def plot_probe_kde_pca_2d(probe, embeddings, labels, emotion_order, title, use_probabilities=True):
        probe_space = get_probe_space(probe, embeddings, use_probabilities=use_probabilities)

        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        X_2d = pca.fit_transform(probe_space)

        print(f"\n{title}")
        print("Probe space shape:", probe_space.shape)
        print("Explained variance ratio:", pca.explained_variance_ratio_.round(4))

        plt.figure(figsize=(9, 7))

        # scatter lightly underneath
        for emotion in emotion_order:
            mask = (np.array(labels) == emotion)
            plt.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                alpha=0.08,
                s=8,
                label=emotion
            )

        # KDE contours if seaborn is available
        try:
            import seaborn as sns

            for emotion in emotion_order:
                mask = (np.array(labels) == emotion)
                sns.kdeplot(
                    x=X_2d[mask, 0],
                    y=X_2d[mask, 1],
                    levels=[0.25, 0.5],
                    linewidths=2
                )
        except ImportError:
            print("seaborn not installed, showing scatter only.")

        plt.title(title)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
        plt.grid(alpha=0.2)
        plt.show()

    # Example prediction
    example_texts = [
        "I am so happy and excited about this!",
        "This is disgusting and makes me sick.",
        "I am terrified something bad will happen.",
        "I miss you so much and feel empty.",
        "What a wonderful surprise!",
        "I am furious and cannot stand this anymore."
    ]

    def predict_examples(text_list, tokenizer, model, device, probe, use_mean=True):
        mean_embs, cls_embs = encode_texts(
            text_list, tokenizer, model, device,
            batch_size=8, max_length=MAX_LENGTH
        )

        embs = mean_embs if use_mean else cls_embs
        embs = normalize(embs, norm="l2")

        preds = probe.predict(embs)
        probs = probe.predict_proba(embs)

        for text, pred, prob in zip(text_list, preds, probs):
            top_idx = np.argsort(prob)[::-1][:3]
            top3 = [(probe.classes_[i], round(float(prob[i]), 3)) for i in top_idx]

            print("\nTEXT:", text)
            print("Predicted:", pred)
            print("Top-3:", top3)


    # Probe weight analysis

    def get_probe_weight_matrix(probe):
        """
        Returns normalized probe weight vectors.
        Shape: (n_classes, hidden_dim)
        """
        if not hasattr(probe, "coef_"):
            raise ValueError("Probe does not expose coef_. This works for LogisticRegression.")
        W = probe.coef_.copy()
        return normalize(W, norm="l2")


    def plot_probe_weight_cosine_similarity(probe, class_order, title):
        W = get_probe_weight_matrix(probe)
        sim = cosine_similarity(W)
        sim_df = pd.DataFrame(sim, index=class_order, columns=class_order).round(3)

        print(f"\n{title}")
        print(sim_df)

        plt.figure(figsize=(7, 6))
        plt.imshow(sim, interpolation="nearest")
        plt.xticks(range(len(class_order)), class_order, rotation=45, ha="right")
        plt.yticks(range(len(class_order)), class_order)
        plt.colorbar()
        plt.title(title)
        plt.tight_layout()
        plt.show()

        return sim_df


    def plot_probe_weights_pca_2d(probe, class_order, title):
        """
        PCA on probe weight vectors themselves.
        Each point = one emotion direction.
        """
        W = get_probe_weight_matrix(probe)

        # with few classes, max possible rank is limited
        n_comp = min(2, W.shape[0], W.shape[1])
        pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
        W_2d = pca.fit_transform(W)

        print(f"\n{title}")
        print("Weight matrix shape:", W.shape)
        print("Explained variance ratio:", pca.explained_variance_ratio_.round(4))

        plt.figure(figsize=(8, 6))
        for i, emotion in enumerate(class_order):
            plt.scatter(W_2d[i, 0], W_2d[i, 1], s=180)
            plt.text(W_2d[i, 0], W_2d[i, 1], emotion, fontsize=11)

        plt.title(title)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(alpha=0.25)
        plt.show()

        return W_2d, pca


    def plot_probe_weights_pca_3d(probe, class_order, title):
        W = get_probe_weight_matrix(probe)

        n_comp = min(3, W.shape[0], W.shape[1])
        if n_comp < 3:
            print(f"\n{title}")
            print("Not enough dimensions/classes for 3D PCA.")
            return None, None

        pca = PCA(n_components=3, random_state=RANDOM_STATE)
        W_3d = pca.fit_transform(W)

        print(f"\n{title}")
        print("Weight matrix shape:", W.shape)
        print("Explained variance ratio:", pca.explained_variance_ratio_.round(4))

        print("\n3D PCA coordinates (per emotion):")
        coords_df = pd.DataFrame(W_3d, columns=["PC1", "PC2", "PC3"])
        coords_df["emotion"] = class_order
        print(coords_df.round(4).to_string(index=False))

        print("\nDesmos-friendly format:")
        for i, emotion in enumerate(class_order):
            x, y, z = W_3d[i]
            print(f"{emotion}: ({x:.4f}, {y:.4f}, {z:.4f})")

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        for i, emotion in enumerate(class_order):
            ax.scatter(W_3d[i, 0], W_3d[i, 1], W_3d[i, 2], s=160)
            ax.text(W_3d[i, 0], W_3d[i, 1], W_3d[i, 2], emotion, fontsize=10)

        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        plt.show()

        return W_3d, pca


    def plot_probe_weights_vs_centroids_pca_2d(probe, centroids, class_order, title):
        """
        Compare probe weight vectors and class centroids in the same PCA space.
        PCA is fit on the combined matrix [weights; centroids].
        """
        W = get_probe_weight_matrix(probe)
        C = normalize(centroids.copy(), norm="l2")

        combined = np.vstack([W, C])

        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        combined_2d = pca.fit_transform(combined)

        W_2d = combined_2d[:len(class_order)]
        C_2d = combined_2d[len(class_order):]

        print(f"\n{title}")
        print("Combined shape:", combined.shape)
        print("Explained variance ratio:", pca.explained_variance_ratio_.round(4))

        plt.figure(figsize=(9, 7))

        for i, emotion in enumerate(class_order):
            # weights
            plt.scatter(W_2d[i, 0], W_2d[i, 1], marker="^", s=180)
            plt.text(W_2d[i, 0], W_2d[i, 1], f"{emotion}_W", fontsize=10)

            # centroids
            plt.scatter(C_2d[i, 0], C_2d[i, 1], marker="o", s=180)
            plt.text(C_2d[i, 0], C_2d[i, 1], f"{emotion}_C", fontsize=10)

            # line connecting them
            plt.plot(
                [W_2d[i, 0], C_2d[i, 0]],
                [W_2d[i, 1], C_2d[i, 1]],
                linestyle="--",
                alpha=0.5
            )

        plt.title(title)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(alpha=0.25)
        plt.show()

        return W_2d, C_2d, pca


    def plot_probe_weight_projection_histograms(probe, embeddings, labels, emotion_order, title_prefix):
        """
        For each emotion direction, project all embeddings onto that weight vector.
        This shows whether the corresponding class tends to score higher along its own direction.
        """
        W = get_probe_weight_matrix(probe)
        X = normalize(embeddings.copy(), norm="l2")
        label_array = np.array(labels)

        for i, emotion in enumerate(emotion_order):
            scores = X @ W[i]

            plt.figure(figsize=(8, 5))
            for e in emotion_order:
                mask = label_array == e
                plt.hist(scores[mask], bins=40, alpha=0.35, label=e, density=True)

            plt.title(f"{title_prefix}: projection onto '{emotion}' weight")
            plt.xlabel("Cosine-like projection score")
            plt.ylabel("Density")
            plt.legend()
            plt.grid(alpha=0.2)
            plt.show()


    # Projection-based classifiers

    def get_probe_weight_matrix(probe):
        """
        Returns normalized probe weight vectors.
        Shape: (n_classes, hidden_dim)
        """
        W = probe.coef_.copy()
        return normalize(W, norm="l2")


    def predict_with_weight_cosine(embeddings, probe, class_order):
        """
        Classify by cosine similarity between embeddings and probe weight vectors.
        """
        X = normalize(embeddings.copy(), norm="l2")
        W = get_probe_weight_matrix(probe)  # already normalized

        scores = X @ W.T   # cosine-like similarity since both normalized
        pred_idx = np.argmax(scores, axis=1)
        preds = np.array([class_order[i] for i in pred_idx])

        return preds, scores


    def predict_with_prototypes(embeddings, centroids, class_order):
        """
        Classify by nearest normalized class centroid (cosine similarity).
        """
        X = normalize(embeddings.copy(), norm="l2")
        C = normalize(centroids.copy(), norm="l2")

        scores = X @ C.T
        pred_idx = np.argmax(scores, axis=1)
        preds = np.array([class_order[i] for i in pred_idx])

        return preds, scores


    def softmax_np(x):
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


    def evaluate_simple_classifier(gold_labels, preds, scores, class_order, name="Classifier"):
        """
        Evaluation for cosine / prototype classifiers.
        Uses softmax(scores) as pseudo-probabilities for log loss + confidence stats.
        """
        probs = softmax_np(scores)

        print(f"\n{name} on XED test")
        print("Accuracy:", round(accuracy_score(gold_labels, preds), 4))
        print("Balanced accuracy:", round(balanced_accuracy_score(gold_labels, preds), 4))
        print("Macro F1:", round(f1_score(gold_labels, preds, average="macro"), 4))
        print("Weighted F1:", round(f1_score(gold_labels, preds, average="weighted"), 4))
        print("Log loss:", round(log_loss(gold_labels, probs, labels=class_order), 4))

        print("\nClassification report:")
        print(classification_report(gold_labels, preds, digits=3))

        cm = confusion_matrix(gold_labels, preds, labels=class_order)
        cm_df = pd.DataFrame(cm, index=class_order, columns=class_order)

        print("Confusion matrix:")
        print(cm_df.to_string())

        pred_idx = np.argmax(probs, axis=1)
        pred_conf = probs[np.arange(len(probs)), pred_idx]

        confidence_df = pd.DataFrame({
            "gold": gold_labels,
            "pred": preds,
            "confidence": pred_conf
        })

        per_class_conf = (
            confidence_df.groupby("gold")["confidence"]
            .agg(["mean", "std", "count"])
            .round(4)
            .rename_axis("emotion")
            .reset_index()
        )

        print("\nPer-class confidence stats:")
        print(per_class_conf.to_string(index=False))

        return {
            "preds": preds,
            "scores": scores,
            "probs": probs,
            "confusion_matrix": cm_df,
            "confidence_stats": per_class_conf,
        }

    def predict_with_linear_scores(embeddings, probe, class_order, normalize_inputs=False):
        """
        Classify using the probe's raw linear scoring function

        If normalize_inputs=True, embeddings are L2-normalized first.
        This is still interpretable, but closer to the original logistic probe
        than cosine similarity to normalized weights.
        """
        X = embeddings.copy()
        if normalize_inputs:
            X = normalize(X, norm="l2")

        W = probe.coef_
        b = probe.intercept_

        scores = X @ W.T + b
        pred_idx = np.argmax(scores, axis=1)
        preds = np.array([class_order[i] for i in pred_idx])

        return preds, scores

    import numpy as np
    import pandas as pd

    from scipy.optimize import minimize
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, normalize
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        classification_report,
        confusion_matrix,
        log_loss
    )


    # Helpers

    def softmax_np(scores):
        scores = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)


    def fit_bias_only(X_calib, y_calib, probe, class_order, l2=0.01):
        """
        Freeze probe weights W, learn only bias b on calibration data.
        """
        W = probe.coef_.copy()
        old_b = probe.intercept_.copy()

        label_to_idx = {label: i for i, label in enumerate(class_order)}
        y_idx = np.array([label_to_idx[y] for y in y_calib])

        def objective(b):
            scores = X_calib @ W.T + b
            probs = softmax_np(scores)

            nll = -np.mean(np.log(probs[np.arange(len(y_idx)), y_idx] + 1e-12))

            # regularize toward original bias, so calibration does not overfit too much
            penalty = l2 * np.sum((b - old_b) ** 2)

            return nll + penalty

        result = minimize(
            objective,
            x0=old_b,
            method="BFGS"
        )

        return result.x


    def predict_with_fixed_weights_new_bias(X, probe, new_b, class_order):
        W = probe.coef_.copy()
        scores = X @ W.T + new_b
        probs = softmax_np(scores)

        pred_idx = np.argmax(probs, axis=1)
        preds = np.array([class_order[i] for i in pred_idx])

        return preds, probs, scores


    def evaluate_bias_calibrated(X_test, y_test, probe, new_b, class_order, name):
        preds, probs, scores = predict_with_fixed_weights_new_bias(
            X_test,
            probe,
            new_b,
            class_order
        )


        # Top-3 accuracy

        top_k = 3

        # get indices of top-3 predictions per sample
        top_k_preds = np.argsort(probs, axis=1)[:, -top_k:]

        # map labels to indices
        label_to_idx = {label: i for i, label in enumerate(class_order)}
        gold_idx = np.array([label_to_idx[y] for y in y_test])

        # check if gold label is in top-k
        top_k_correct = [
            gold_idx[i] in top_k_preds[i]
            for i in range(len(gold_idx))
        ]

        top_k_acc = np.mean(top_k_correct)

        print(f"Top-{top_k} accuracy:", round(top_k_acc, 4))

        print(f"\n{name}")
        print("Accuracy:", round(accuracy_score(y_test, preds), 4))
        print("Balanced accuracy:", round(balanced_accuracy_score(y_test, preds), 4))
        print("Macro F1:", round(f1_score(y_test, preds, average="macro"), 4))
        print("Weighted F1:", round(f1_score(y_test, preds, average="weighted"), 4))
        print("Log loss:", round(log_loss(y_test, probs, labels=class_order), 4))

        print("\nClassification report:")
        print(classification_report(y_test, preds, digits=3))

        cm = confusion_matrix(y_test, preds, labels=class_order)
        cm_df = pd.DataFrame(cm, index=class_order, columns=class_order)

        print("Confusion matrix:")
        print(cm_df.to_string())

        return {
            "preds": preds,
            "probs": probs,
            "scores": scores,
            "confusion_matrix": cm_df
        }

    # Model loading !!!

    MODEL_NAME = "google-bert/bert-base-multilingual-cased"
    # MODEL_NAME = "FacebookAI/xlm-roberta-base"
    # MODEL_NAME = "microsoft/mdeberta-v3-base"

    df = pd.read_csv(DATA_FILE)
    df["text"] = df["text"].astype(str)
    df["emotion"] = df["emotion"].astype(str)

    texts = df["text"].tolist()
    labels = df["emotion"].tolist()
    emotion_order = sorted(df["emotion"].unique())

    print("Rows:", len(df))
    print("Emotions:", emotion_order)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    # mBERT info

    # embeddings
    mean_embeddings, cls_embeddings = encode_texts(
        texts, tokenizer, model, device,
        batch_size=BATCH_SIZE, max_length=MAX_LENGTH
    )

    print("Mean embeddings shape:", mean_embeddings.shape)
    print("CLS embeddings shape:", cls_embeddings.shape)

    # normalization
    mean_embeddings_norm = normalize(mean_embeddings, norm="l2")
    cls_embeddings_norm = normalize(cls_embeddings, norm="l2")  

    # centroids
    mean_centroids = compute_centroids(mean_embeddings_norm, df, emotion_order)
    cls_centroids = compute_centroids(cls_embeddings_norm, df, emotion_order)

    # cos similarity
    mean_centroid_cos = cosine_similarity(mean_centroids)
    cls_centroid_cos = cosine_similarity(cls_centroids)

    print("\nCosine similarity between mean-pooled centroids:")
    print(pd.DataFrame(mean_centroid_cos, index=emotion_order, columns=emotion_order).round(3))

    print("\nCosine similarity between CLS centroids:")
    print(pd.DataFrame(cls_centroid_cos, index=emotion_order, columns=emotion_order).round(3))



    # mBERT visualizations

    plot_centroids_in_full_pca_2d(
        mean_embeddings_norm,
        mean_centroids,
        emotion_order,
        "Emotion centroids projected into full mean-pooled PCA space"
    )

    plot_centroids_in_full_pca_3d(
        mean_embeddings_norm,
        mean_centroids,
        emotion_order,
        "Emotion centroids in 3D PCA space (mean pooled, PCA fit on full data)"
    )

    # Optional direct comparison with CLS
    plot_centroids_in_full_pca_2d(
        cls_embeddings_norm,
        cls_centroids,
        emotion_order,
        "Emotion centroids projected into full CLS PCA space"
    )

    # probe training

    mean_probe = train_probe(mean_embeddings_norm, labels, "Mean-pooled")
    cls_probe = train_probe(cls_embeddings_norm, labels, "CLS")

    # probe visualization

    plot_probe_pca_2d(
        mean_probe,
        mean_embeddings_norm,
        labels,
        emotion_order,
        "PCA of mean-pooled linear probe decision space",
        use_probabilities=False
    )

    plot_probe_pca_2d(
        mean_probe,
        mean_embeddings_norm,
        labels,
        emotion_order,
        "PCA of mean-pooled linear probe probability space",
        use_probabilities=True
    )

    plot_probe_centroids_pca_2d(
        mean_probe,
        mean_embeddings_norm,
        df,
        emotion_order,
        "Emotion centroids in mean-pooled linear probe decision space",
        use_probabilities=False
    )

    plot_probe_centroids_pca_2d(
        mean_probe,
        mean_embeddings_norm,
        df,
        emotion_order,
        "Emotion centroids in mean-pooled linear probe probability space",
        use_probabilities=True
    )

    plot_probe_kde_pca_2d(
        mean_probe,
        mean_embeddings_norm,
        labels,
        emotion_order,
        "KDE contours of mean-pooled linear probe probability space",
        use_probabilities=True
    )


    # probe visualization

    plot_probe_pca_2d(
        cls_probe,
        cls_embeddings_norm,
        labels,
        emotion_order,
        "PCA of CLS linear probe decision space",
        use_probabilities=False
    )

    plot_probe_pca_2d(
        cls_probe,
        cls_embeddings_norm,
        labels,
        emotion_order,
        "PCA of CLS linear probe probability space",
        use_probabilities=True
    )

    plot_probe_centroids_pca_2d(
        cls_probe,
        cls_embeddings_norm,
        df,
        emotion_order,
        "Emotion centroids in CLS linear probe decision space",
        use_probabilities=False
    )

    plot_probe_centroids_pca_2d(
        cls_probe,
        cls_embeddings_norm,
        df,
        emotion_order,
        "Emotion centroids in CLS linear probe probability space",
        use_probabilities=True
    )

    plot_probe_kde_pca_2d(
        cls_probe,
        cls_embeddings_norm,
        labels,
        emotion_order,
        "KDE contours of CLS linear probe probability space",
        use_probabilities=True
    )



    # Probe weight analysis

    class_order = list(mean_probe.classes_)

    # Mean-pooled probe weights
    plot_probe_weight_cosine_similarity(
        mean_probe,
        class_order,
        "Cosine similarity between mean-pooled probe weight vectors"
    )

    plot_probe_weights_pca_2d(
        mean_probe,
        class_order,
        "PCA of mean-pooled probe weight vectors"
    )

    plot_probe_weights_pca_3d(
        mean_probe,
        class_order,
        "3D PCA of mean-pooled probe weight vectors"
    )

    plot_probe_weights_vs_centroids_pca_2d(
        mean_probe,
        mean_centroids,
        class_order,
        "Mean-pooled probe weights vs mean-pooled class centroids"
    )


    # CLS probe weights
    plot_probe_weight_cosine_similarity(
        cls_probe,
        class_order,
        "Cosine similarity between CLS probe weight vectors"
    )

    plot_probe_weights_pca_2d(
        cls_probe,
        class_order,
        "PCA of CLS probe weight vectors"
    )

    plot_probe_weights_pca_3d(
        cls_probe,
        class_order,
        "3D PCA of CLS probe weight vectors"
    )

    plot_probe_weights_vs_centroids_pca_2d(
        cls_probe,
        cls_centroids,
        class_order,
        "CLS probe weights vs CLS class centroids"
    )

    # example prediction and t-sne


    print("\nExample predictions with mean-pooled probe")
    predict_examples(example_texts, tokenizer, model, device, mean_probe, use_mean=True)

    print("\nExample predictions with CLS probe")
    predict_examples(example_texts, tokenizer, model, device, cls_probe, use_mean=False)

    X_tsne_input = mean_embeddings_norm
    y_tsne = df["emotion"].values

    from datasets import load_dataset
    import pandas as pd

    target_emotions = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
    random_state = 42

    dfs = []

    for split in ["train", "dev", "test"]:
        ds = load_dataset(
            "brighter-dataset/BRIGHTER-emotion-categories",
            "arq",
            split=split
        )
        df_part = ds.to_pandas()
        dfs.append(df_part)

    brighter_df = pd.concat(dfs, ignore_index=True)
    brighter_df["text"] = brighter_df["text"].astype(str)

    # ensure all emotion columns exist + clean
    for col in target_emotions:
        if col not in brighter_df.columns:
            brighter_df[col] = 0
        brighter_df[col] = brighter_df[col].fillna(0).astype(int)

    # keep only single-label
    brighter_df["n_active"] = brighter_df[target_emotions].sum(axis=1)
    brighter_df = brighter_df[brighter_df["n_active"] == 1].copy()

    # convert to single label
    brighter_df["emotion"] = brighter_df[target_emotions].idxmax(axis=1)

    brighter_df = brighter_df[["text", "emotion"]]

    print("\nBRIGHTER distribution:")
    print(brighter_df["emotion"].value_counts())


    # second arabic

    for split in ["train", "dev", "test"]:
        ds = load_dataset(
            "brighter-dataset/BRIGHTER-emotion-categories",
            "ary",
            split=split
        )
        df_part = ds.to_pandas()
        dfs.append(df_part)

    brighter_df_2 = pd.concat(dfs, ignore_index=True)
    brighter_df_2["text"] = brighter_df_2["text"].astype(str)

    # ensure all emotion columns exist + clean
    for col in target_emotions:
        if col not in brighter_df_2.columns:
            brighter_df_2[col] = 0
        brighter_df_2[col] = brighter_df_2[col].fillna(0).astype(int)

    # keep only single-label
    brighter_df_2["n_active"] = brighter_df_2[target_emotions].sum(axis=1)
    brighter_df_2 = brighter_df_2[brighter_df_2["n_active"] == 1].copy()

    # convert to single label
    brighter_df_2["emotion"] = brighter_df_2[target_emotions].idxmax(axis=1)

    brighter_df_2 = brighter_df_2[["text", "emotion"]]

    print("\nBRIGHTER distribution 2:")
    print(brighter_df_2["emotion"].value_counts())


    # go_df = pd.read_csv("goemotions_German.csv")

    # go_df["text"] = go_df["text"].astype(str)
    # go_df["emotion"] = go_df["emotion"].astype(str)

    # go_df = go_df[go_df["emotion"].isin(target_emotions)]

    # print("\nGoEmotions distribution:")
    # print(go_df["emotion"].value_counts())


    # 2.1 LOAD XED (your CSV)
    xed_df = pd.read_csv("xed_arabic.csv")

    xed_df["text"] = xed_df["text"].astype(str)
    xed_df["emotion"] = xed_df["emotion"].astype(str)

    # keep only target emotions (important!)
    xed_df = xed_df[xed_df["emotion"].isin(target_emotions)]

    print("XED distribution:")
    print(xed_df["emotion"].value_counts())


    comb_df = pd.concat([brighter_df, brighter_df_2, xed_df], ignore_index=True)

    print("\nCombined distribution BEFORE balancing:")
    print(comb_df["emotion"].value_counts())


    target_n = comb_df["emotion"].value_counts().min()

    balanced_parts = []

    for emotion, group in comb_df.groupby("emotion"):
        balanced_parts.append(
            group.sample(n=target_n, random_state=random_state)
        )

    comb_df = (
        pd.concat(balanced_parts, ignore_index=True)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    print("\nBalanced rows:", len(comb_df))
    print("\nBalanced distribution:")
    print(comb_df["emotion"].value_counts())


    xed_texts = comb_df["text"].tolist()
    xed_labels = comb_df["emotion"].tolist()


    xed_mean_embeddings, xed_cls_embeddings = encode_texts(
        xed_texts,
        tokenizer,
        model,
        device,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH
    )

    xed_mean_embeddings_norm = normalize(xed_mean_embeddings, norm="l2")
    xed_cls_embeddings_norm = normalize(xed_cls_embeddings, norm="l2")

    print("\nXED mean embeddings shape:", xed_mean_embeddings_norm.shape)
    print("XED CLS embeddings shape:", xed_cls_embeddings_norm.shape)

    print("\nMean probe classes:", mean_probe.classes_)
    print("CLS probe classes:", cls_probe.classes_)

    class_order = list(mean_probe.classes_)

    # Split df into calibration and test
    mean_X_calib, mean_X_test, y_calib, y_test = train_test_split(
        xed_mean_embeddings_norm,
        xed_labels,
        test_size=0.5,
        random_state=42,
        stratify=xed_labels
    )

    cls_X_calib, cls_X_test, _, _ = train_test_split(
        xed_cls_embeddings_norm,
        xed_labels,
        test_size=0.5,
        random_state=42,
        stratify=xed_labels
    )



    print("\nCalibration set size:", len(y_calib))
    print("Test set size:", len(y_test))



    # Evaluation stats
    def evaluate_probe_on_single_label(embeddings, probe, gold_labels, class_order, name="Probe"):
        preds = probe.predict(embeddings)
        probs = probe.predict_proba(embeddings)

        print(f"\n{name} on XED test")
        print("Accuracy:", round(accuracy_score(gold_labels, preds), 4))
        print("Balanced accuracy:", round(balanced_accuracy_score(gold_labels, preds), 4))
        print("Macro F1:", round(f1_score(gold_labels, preds, average="macro"), 4))
        print("Weighted F1:", round(f1_score(gold_labels, preds, average="weighted"), 4))
        print("Log loss:", round(log_loss(gold_labels, probs, labels=class_order), 4))


        # Top-3 accuracy

        top_k = 3

        # get indices of top-3 predictions per sample
        top_k_preds = np.argsort(probs, axis=1)[:, -top_k:]

        # map labels to indices
        label_to_idx = {label: i for i, label in enumerate(class_order)}
        gold_idx = np.array([label_to_idx[y] for y in gold_labels])

        # check if gold label is in top-k
        top_k_correct = [
            gold_idx[i] in top_k_preds[i]
            for i in range(len(gold_idx))
        ]

        top_k_acc = np.mean(top_k_correct)

        print(f"Top-{top_k} accuracy:", round(top_k_acc, 4))

        print("\nClassification report:")
        print(classification_report(gold_labels, preds, digits=3))

        cm = confusion_matrix(gold_labels, preds, labels=class_order)
        cm_df = pd.DataFrame(cm, index=class_order, columns=class_order)

        print("Confusion matrix:")
        print(cm_df.to_string())

        pred_idx = np.argmax(probs, axis=1)
        pred_conf = probs[np.arange(len(probs)), pred_idx]

        confidence_df = pd.DataFrame({
            "gold": gold_labels,
            "pred": preds,
            "confidence": pred_conf
        })

        per_class_conf = (
            confidence_df.groupby("gold")["confidence"]
            .agg(["mean", "std", "count"])
            .round(4)
            .rename_axis("emotion")
            .reset_index()
        )

        print("\nPer-class confidence stats:")
        print(per_class_conf.to_string(index=False))

        return {
            "preds": preds,
            "probs": probs,
            "confusion_matrix": cm_df,
            "confidence_stats": per_class_conf,
        }


    # Baseline: uncalibrated
    mean_base_results = evaluate_probe_on_single_label(
        mean_X_test,
        mean_probe,
        y_test,
        class_order,
        name="Mean-pooled mBERT (uncalibrated)"
    )

    # HUMAN-LABEL FINE-TUNED PROBE

    from sklearn.linear_model import LogisticRegression

    def train_eval_human_probe(X_calib, X_test, y_calib, y_test, class_order, name):
        human_probe = LogisticRegression(
            max_iter=3000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=42
        )

        human_probe.fit(X_calib, y_calib)

        preds = human_probe.predict(X_test)
        probs = human_probe.predict_proba(X_test)

        print(f"\n{name}: probe trained on human calibration data")
        print("Accuracy:", round(accuracy_score(y_test, preds), 4))
        print("Balanced accuracy:", round(balanced_accuracy_score(y_test, preds), 4))
        print("Macro F1:", round(f1_score(y_test, preds, average="macro"), 4))
        print("Weighted F1:", round(f1_score(y_test, preds, average="weighted"), 4))
        print("Log loss:", round(log_loss(y_test, probs, labels=class_order), 4))

        top_k = 3
        top_k_preds = np.argsort(probs, axis=1)[:, -top_k:]
        label_to_idx = {label: i for i, label in enumerate(class_order)}
        gold_idx = np.array([label_to_idx[y] for y in y_test])
        top_k_acc = np.mean([gold_idx[i] in top_k_preds[i] for i in range(len(gold_idx))])
        print(f"Top-{top_k} accuracy:", round(top_k_acc, 4))

        print("\nClassification report:")
        print(classification_report(y_test, preds, digits=3))

        cm = confusion_matrix(y_test, preds, labels=class_order)
        cm_df = pd.DataFrame(cm, index=class_order, columns=class_order)
        print("Confusion matrix:")
        print(cm_df.to_string())

        return {
            "model": human_probe,
            "preds": preds,
            "probs": probs,
            "confusion_matrix": cm_df
        }


    mean_human_probe_results = train_eval_human_probe(
        mean_X_calib,
        mean_X_test,
        y_calib,
        y_test,
        class_order,
        name="Mean-pooled"
    )

    cls_human_probe_results = train_eval_human_probe(
        cls_X_calib,
        cls_X_test,
        y_calib,
        y_test,
        class_order,
        name="CLS"
    )

    from sklearn.linear_model import LogisticRegression

    # 6D EMOTION SUBSPACE CLASSIFIER

    def project_to_probe_subspace(X, probe, use_bias=False):
        W = probe.coef_.copy()
        Z = X @ W.T

        if use_bias:
            Z = Z + probe.intercept_

        return Z


    def train_eval_6d_classifier(X_calib, X_test, y_calib, y_test, probe, class_order, name, use_bias=False):
        Z_calib = project_to_probe_subspace(X_calib, probe, use_bias=use_bias)
        Z_test = project_to_probe_subspace(X_test, probe, use_bias=use_bias)

        clf_6d = LogisticRegression(
            max_iter=3000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=42
        )

        clf_6d.fit(Z_calib, y_calib)

        preds = clf_6d.predict(Z_test)
        probs = clf_6d.predict_proba(Z_test)

        print(f"\n{name}: Logistic Regression on 6D probe subspace")
        print("6D train shape:", Z_calib.shape)
        print("6D test shape:", Z_test.shape)
        print("Accuracy:", round(accuracy_score(y_test, preds), 4))
        print("Balanced accuracy:", round(balanced_accuracy_score(y_test, preds), 4))
        print("Macro F1:", round(f1_score(y_test, preds, average="macro"), 4))
        print("Weighted F1:", round(f1_score(y_test, preds, average="weighted"), 4))
        print("Log loss:", round(log_loss(y_test, probs, labels=class_order), 4))

        print("\nClassification report:")
        print(classification_report(y_test, preds, digits=3))

        cm = confusion_matrix(y_test, preds, labels=class_order)
        cm_df = pd.DataFrame(cm, index=class_order, columns=class_order)

        print("Confusion matrix:")
        print(cm_df.to_string())

        return {
            "model": clf_6d,
            "Z_calib": Z_calib,
            "Z_test": Z_test,
            "preds": preds,
            "probs": probs,
            "confusion_matrix": cm_df
        }


    mean_6d_results = train_eval_6d_classifier(
        mean_X_calib,
        mean_X_test,
        y_calib,
        y_test,
        mean_probe,
        class_order,
        name="Mean-pooled"
    )

    cls_6d_results = train_eval_6d_classifier(
        cls_X_calib,
        cls_X_test,
        y_calib,
        y_test,
        cls_probe,
        class_order,
        name="CLS"
    )

    mean_6d_human_probe_results = train_eval_6d_classifier(
        mean_X_calib,
        mean_X_test,
        y_calib,
        y_test,
        mean_human_probe_results["model"],
        class_order,
        name="Mean-pooled human-probe 6D"
    )

    cls_6d_human_probe_results = train_eval_6d_classifier(
        cls_X_calib,
        cls_X_test,
        y_calib,
        y_test,
        cls_human_probe_results["model"],
        class_order,
        name="CLS human-probe 6D"
    )

    # LABEL DISAGREEMENT / ERROR INSPECTION

    # Recreate split with indices so we can recover the original texts
    all_indices = np.arange(len(comb_df))

    mean_idx_calib, mean_idx_test, _, _ = train_test_split(
        all_indices,
        xed_labels,
        test_size=0.5,
        random_state=42,
        stratify=xed_labels
    )

    test_texts = comb_df.iloc[mean_idx_test]["text"].tolist()
    test_gold = np.array(y_test)

    def inspect_strong_disagreements(X_test, probe, test_texts, gold_labels, class_order, n=30, name="Probe"):
        probs = probe.predict_proba(X_test)
        preds = probe.predict(X_test)

        pred_idx = np.argmax(probs, axis=1)
        pred_conf = probs[np.arange(len(probs)), pred_idx]

        rows = []

        for i in range(len(gold_labels)):
            if preds[i] != gold_labels[i]:
                top3_idx = np.argsort(probs[i])[::-1][:3]
                top3 = [(class_order[j], round(float(probs[i, j]), 3)) for j in top3_idx]

                rows.append({
                    "text": test_texts[i],
                    "gold": gold_labels[i],
                    "pred": preds[i],
                    "confidence": pred_conf[i],
                    "top3": top3
                })

        err_df = pd.DataFrame(rows).sort_values("confidence", ascending=False)

        print(f"\nStrongest disagreements: {name}")
        print("Total errors:", len(err_df))

        for _, row in err_df.head(n).iterrows():
            print("\nTEXT:", row["text"])
            print("Gold:", row["gold"])
            print("Pred:", row["pred"])
            print("Confidence:", round(row["confidence"], 4))
            print("Top-3:", row["top3"])

        return err_df


    mean_error_df = inspect_strong_disagreements(
        mean_X_test,
        mean_probe,
        test_texts,
        test_gold,
        class_order,
        n=30,
        name="Mean-pooled probe"
    )

    cls_error_df = inspect_strong_disagreements(
        cls_X_test,
        cls_probe,
        test_texts,
        test_gold,
        class_order,
        n=30,
        name="CLS probe"
    )


    # Projection classifier: cosine to probe weights

    mean_weight_preds, mean_weight_scores = predict_with_weight_cosine(
        mean_X_test,
        mean_probe,
        class_order
    )

    mean_weight_results = evaluate_simple_classifier(
        y_test,
        mean_weight_preds,
        mean_weight_scores,
        class_order,
        name="Mean-pooled XLM-R (cosine to probe weights)"
    )

    cls_weight_preds, cls_weight_scores = predict_with_weight_cosine(
        cls_X_test,
        cls_probe,
        class_order
    )

    cls_weight_results = evaluate_simple_classifier(
        y_test,
        cls_weight_preds,
        cls_weight_scores,
        class_order,
        name="CLS XLM-R (cosine to probe weights)"
    )


    # Prototype classifier: nearest centroid

    mean_proto_preds, mean_proto_scores = predict_with_prototypes(
        mean_X_test,
        mean_centroids,
        class_order
    )

    mean_proto_results = evaluate_simple_classifier(
        y_test,
        mean_proto_preds,
        mean_proto_scores,
        class_order,
        name="Mean-pooled XLM-R (prototype classifier)"
    )

    cls_proto_preds, cls_proto_scores = predict_with_prototypes(
        cls_X_test,
        cls_centroids,
        class_order
    )

    cls_proto_results = evaluate_simple_classifier(
        y_test,
        cls_proto_preds,
        cls_proto_scores,
        class_order,
        name="CLS XLM-R (prototype classifier)"
    )




    example_texts = [
        "I just got promoted at work, I can't believe it!",
        "This is absolutely disgusting, I feel sick.",
        "I'm really scared about what might happen tomorrow.",
        "I miss you so much, everything feels empty without you.",
        "Wow, I did not expect that at all!"
    ]

    def print_topk_linear_predictions(
        texts,
        tokenizer,
        model,
        device,
        probe,
        class_order,
        use_mean=True,
        top_k=3
    ):
        # Encode
        mean_embs, cls_embs = encode_texts(
            texts,
            tokenizer,
            model,
            device,
            batch_size=8,
            max_length=MAX_LENGTH
        )

        if use_mean:
            X = normalize(mean_embs, norm="l2")
        else:
            X = normalize(cls_embs, norm="l2")

        # Get linear scores
        _, scores = predict_with_linear_scores(
            X,
            probe,
            class_order,
            normalize_inputs=False
        )

        print("\nTop-k predictions (linear scores x*w + b)\n")

        for i, text in enumerate(texts):
            score_vec = scores[i]

            # sort descending
            top_indices = np.argsort(score_vec)[::-1][:top_k]

            print(f"Text: {text}")
            for rank, idx in enumerate(top_indices, 1):
                emotion = class_order[idx]
                score = score_vec[idx]
                print(f"  {rank}. {emotion:10s} | score = {score:.4f}")
            print()

        _, scores = predict_with_weight_cosine(
            X,
            probe,
            class_order
        )

        print("\nTop-k predictions (just linear scores x*w)\n")

        for i, text in enumerate(texts):
            score_vec = scores[i]

            # sort descending
            top_indices = np.argsort(score_vec)[::-1][:top_k]

            print(f"Text: {text}")
            for rank, idx in enumerate(top_indices, 1):
                emotion = class_order[idx]
                score = score_vec[idx]
                print(f"  {rank}. {emotion:10s} | score = {score:.4f}")
            print()

    print_topk_linear_predictions(
        example_texts,
        tokenizer,
        model,
        device,
        mean_probe,
        class_order,
        use_mean=True
    )

    def compute_label_centroids(X, y, class_order):
        y = np.array(y)
        centroids = []
        for emotion in class_order:
            mask = y == emotion
            centroids.append(X[mask].mean(axis=0))
        return normalize(np.vstack(centroids), norm="l2")


    # Human-labeled English centroids

    mean_human_centroids = compute_label_centroids(
        mean_X_calib,
        y_calib,
        class_order
    )

    import pandas as pd
    import numpy as np

    from scipy.linalg import orthogonal_procrustes
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import normalize
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        classification_report,
        confusion_matrix,
        log_loss
    )


    # Helpers

    def softmax_np(scores):
        scores = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)


    def evaluate_probe_on_single_label(embeddings, probe, gold_labels, class_order, name="Probe"):
        preds = probe.predict(embeddings)
        probs = probe.predict_proba(embeddings)

        print(f"\n{name}")
        print("Accuracy:", round(accuracy_score(gold_labels, preds), 4))
        print("Balanced accuracy:", round(balanced_accuracy_score(gold_labels, preds), 4))
        print("Macro F1:", round(f1_score(gold_labels, preds, average="macro"), 4))
        print("Weighted F1:", round(f1_score(gold_labels, preds, average="weighted"), 4))
        print("Log loss:", round(log_loss(gold_labels, probs, labels=class_order), 4))

        print("\nClassification report:")
        print(classification_report(gold_labels, preds, digits=3))

        cm = confusion_matrix(gold_labels, preds, labels=class_order)
        cm_df = pd.DataFrame(cm, index=class_order, columns=class_order)

        print("Confusion matrix:")
        print(cm_df.to_string())

        return {
            "preds": preds,
            "probs": probs,
            "confusion_matrix": cm_df
        }


    from scipy.optimize import minimize
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    import numpy as np

    def fit_bias_only(
        X_calib,
        y_calib,
        probe,
        class_order,
        l2_values=(0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0),
        random_state=42,
    ):
        """
        Freeze probe weights W, tune L2 regularization, and learn only bias b.
        L2 is selected using an internal split of the calibration set.
        """
        W = probe.coef_.copy()
        old_b = probe.intercept_.copy()

        label_to_idx = {label: i for i, label in enumerate(class_order)}

        X_train, X_val, y_train, y_val = train_test_split(
            X_calib,
            y_calib,
            test_size=0.3,
            random_state=random_state,
            stratify=y_calib,
        )

        y_train_idx = np.array([label_to_idx[y] for y in y_train])

        def softmax_np(scores):
            scores = scores - scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            return exp_scores / exp_scores.sum(axis=1, keepdims=True)

        def fit_for_l2(l2):
            def objective(b):
                scores = X_train @ W.T + b
                probs = softmax_np(scores)

                nll = -np.mean(
                    np.log(probs[np.arange(len(y_train_idx)), y_train_idx] + 1e-12)
                )

                penalty = l2 * np.sum((b - old_b) ** 2)

                return nll + penalty

            result = minimize(
                objective,
                x0=old_b,
                method="BFGS",
            )

            return result.x

        best_l2 = None
        best_f1 = -1
        best_b = None

        print("\nBias-only calibration L2 search:")

        for l2 in l2_values:
            candidate_b = fit_for_l2(l2)

            val_scores = X_val @ W.T + candidate_b
            val_pred_idx = np.argmax(val_scores, axis=1)
            val_preds = np.array([class_order[i] for i in val_pred_idx])

            val_f1 = f1_score(y_val, val_preds, average="macro")

            print(f"  l2={l2:<8} validation_macro_f1={val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_l2 = l2
                best_b = candidate_b

        print(f"Best l2: {best_l2} | validation_macro_f1={best_f1:.4f}")

        # Refit final bias on the full calibration set using best L2
        y_calib_idx = np.array([label_to_idx[y] for y in y_calib])

        def final_objective(b):
            scores = X_calib @ W.T + b
            probs = softmax_np(scores)

            nll = -np.mean(
                np.log(probs[np.arange(len(y_calib_idx)), y_calib_idx] + 1e-12)
            )

            penalty = best_l2 * np.sum((b - old_b) ** 2)

            return nll + penalty

        final_result = minimize(
            final_objective,
            x0=old_b,
            method="BFGS",
        )

        return final_result.x


    def evaluate_fixed_weights_new_bias(X_test, y_test, probe, new_b, class_order, name):
        W = probe.coef_.copy()
        scores = X_test @ W.T + new_b
        probs = softmax_np(scores)

        pred_idx = np.argmax(probs, axis=1)
        preds = np.array([class_order[i] for i in pred_idx])

        print(f"\n{name}")
        print("Accuracy:", round(accuracy_score(y_test, preds), 4))
        print("Balanced accuracy:", round(balanced_accuracy_score(y_test, preds), 4))
        print("Macro F1:", round(f1_score(y_test, preds, average="macro"), 4))
        print("Weighted F1:", round(f1_score(y_test, preds, average="weighted"), 4))
        print("Log loss:", round(log_loss(y_test, probs, labels=class_order), 4))

        print("\nClassification report:")
        print(classification_report(y_test, preds, digits=3))

        cm = confusion_matrix(y_test, preds, labels=class_order)
        cm_df = pd.DataFrame(cm, index=class_order, columns=class_order)
        print("Confusion matrix:")
        print(cm_df.to_string())

        return {
            "preds": preds,
            "probs": probs,
            "confusion_matrix": cm_df
        }


    def compute_label_centroids(X, y, class_order):
        y = np.array(y)
        centroids = []
        for emotion in class_order:
            mask = y == emotion
            centroids.append(X[mask].mean(axis=0))
        return normalize(np.vstack(centroids), norm="l2")


    def fit_procrustes_from_centroids(X_calib, y_calib, target_centroids, class_order):
        """
        Learns an orthogonal rotation from German calibration centroids
        to English weak-label centroids.
        """
        source_centroids = compute_label_centroids(X_calib, y_calib, class_order)

        R, scale = orthogonal_procrustes(source_centroids, target_centroids)

        print("\nProcrustes scale:", round(scale, 4))

        return R


    def apply_rotation(X, R):
        return normalize(X @ R, norm="l2")

    from sklearn.linear_model import LogisticRegression

    def project_to_probe_subspace(X, probe):
        """
        Projects embeddings into the 6D English probe-weight emotion space:
            Z = X @ W.T
        """
        W = probe.coef_.copy()
        return X @ W.T


    def evaluate_6d_logreg(Z_calib, Z_test, y_calib, y_test, class_order, name):
        clf_6d = LogisticRegression(
            max_iter=3000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=42
        )

        clf_6d.fit(Z_calib, y_calib)

        preds = clf_6d.predict(Z_test)
        probs = clf_6d.predict_proba(Z_test)

        print(f"\n{name}: LogReg on 6D projected emotion space")
        print("6D train shape:", Z_calib.shape)
        print("6D test shape:", Z_test.shape)
        print("Accuracy:", round(accuracy_score(y_test, preds), 4))
        print("Balanced accuracy:", round(balanced_accuracy_score(y_test, preds), 4))
        print("Macro F1:", round(f1_score(y_test, preds, average="macro"), 4))
        print("Weighted F1:", round(f1_score(y_test, preds, average="weighted"), 4))
        print("Log loss:", round(log_loss(y_test, probs, labels=class_order), 4))

        print("\nClassification report:")
        print(classification_report(y_test, preds, digits=3))

        cm = confusion_matrix(y_test, preds, labels=class_order)
        cm_df = pd.DataFrame(cm, index=class_order, columns=class_order)
        print("Confusion matrix:")
        print(cm_df.to_string())

        return {
            "model": clf_6d,
            "preds": preds,
            "probs": probs,
            "confusion_matrix": cm_df
        }

    def evaluate_language(language, paths):
        print(f"\nLanguage: {language}")
        comb_df = pd.concat(map(pd.read_csv, paths), ignore_index=True)
        comb_df["text"] = comb_df["text"].astype(str)
        comb_df["emotion"] = comb_df["emotion"].astype(str)

        print("Original rows:", len(comb_df))
        print("Original class distribution:")
        print(comb_df["emotion"].value_counts())

        target_n = comb_df["emotion"].value_counts().min()
        balanced_parts = [
            group.sample(n=target_n, random_state=RANDOM_STATE)
            for emotion, group in comb_df.groupby("emotion")
        ]
        comb_df = (
            pd.concat(balanced_parts, ignore_index=True)
            .sample(frac=1, random_state=RANDOM_STATE)
            .reset_index(drop=True)
        )
        print("Balanced rows:", len(comb_df))
        print("Balanced class distribution:")
        print(comb_df["emotion"].value_counts())

        german_texts = comb_df["text"].tolist()
        german_labels = comb_df["emotion"].tolist()
        german_mean_embeddings, german_cls_embeddings = encode_texts(
            german_texts, tokenizer, model, device,
            batch_size=BATCH_SIZE, max_length=MAX_LENGTH
        )
        german_mean_embeddings_norm = normalize(german_mean_embeddings, norm="l2")
        german_cls_embeddings_norm = normalize(german_cls_embeddings, norm="l2")
        print("Mean embedding shape:", german_mean_embeddings_norm.shape)
        print("CLS embedding shape:", german_cls_embeddings_norm.shape)

        class_order = list(mean_probe.classes_)
        mean_X_calib, mean_X_test, y_calib, y_test = train_test_split(
            german_mean_embeddings_norm, german_labels,
            test_size=0.5, random_state=RANDOM_STATE, stratify=german_labels
        )
        cls_X_calib, cls_X_test, _, _ = train_test_split(
            german_cls_embeddings_norm, german_labels,
            test_size=0.5, random_state=RANDOM_STATE, stratify=german_labels
        )
        print("Calibration set size:", len(y_calib))
        print("Test set size:", len(y_test))


        mean_ar_direct = evaluate_probe_on_single_label(
            mean_X_test,
            mean_probe,
            y_test,
            class_order,
            name=f"Mean-pooled English probe directly on {language}"
        )

        cls_ar_direct = evaluate_probe_on_single_label(
            cls_X_test,
            cls_probe,
            y_test,
            class_order,
            name=f"CLS English probe directly on {language}"
        )



        mean_ar_bias = fit_bias_only(
            mean_X_calib,
            y_calib,
            mean_probe,
            class_order,

        )

        cls_ar_bias = fit_bias_only(
            cls_X_calib,
            y_calib,
            cls_probe,
            class_order,

        )

        print("\nOriginal mean bias:")
        print(pd.Series(mean_probe.intercept_, index=class_order).round(4))

        print(f"\n{language}-calibrated mean bias:")
        print(pd.Series(mean_ar_bias, index=class_order).round(4))

        print("\nOriginal CLS bias:")
        print(pd.Series(cls_probe.intercept_, index=class_order).round(4))

        print(f"\n{language}-calibrated CLS bias:")
        print(pd.Series(cls_ar_bias, index=class_order).round(4))

        mean_ar_bias_results = evaluate_fixed_weights_new_bias(
            mean_X_test,
            y_test,
            mean_probe,
            mean_ar_bias,
            class_order,
            name=f"Mean-pooled English weights + {language} learned bias"
        )

        cls_ar_bias_results = evaluate_fixed_weights_new_bias(
            cls_X_test,
            y_test,
            cls_probe,
            cls_ar_bias,
            class_order,
            name=f"CLS English weights + {language} learned bias"
        )


        # Direct {language} geometry-only baselines before Procrustes


        mean_direct_weight_preds, mean_direct_weight_scores = predict_with_weight_cosine(
            mean_X_test,
            mean_probe,
            class_order
        )

        mean_direct_weight_results = evaluate_simple_classifier(
            y_test,
            mean_direct_weight_preds,
            mean_direct_weight_scores,
            class_order,
            name=f"Mean-pooled {language} direct (cosine to English probe weights)"
        )

        mean_direct_proto_preds, mean_direct_proto_scores = predict_with_prototypes(
            mean_X_test,
            mean_centroids,
            class_order
        )

        mean_direct_proto_results = evaluate_simple_classifier(
            y_test,
            mean_direct_proto_preds,
            mean_direct_proto_scores,
            class_order,
            name=f"Mean-pooled {language} direct (nearest English weak centroid)"
        )

        # CLS

        cls_direct_weight_preds, cls_direct_weight_scores = predict_with_weight_cosine(
            cls_X_test,
            cls_probe,
            class_order
        )

        cls_direct_weight_results = evaluate_simple_classifier(
            y_test,
            cls_direct_weight_preds,
            cls_direct_weight_scores,
            class_order,
            name=f"CLS {language} direct (cosine to Arabic probe weights)"
        )

        cls_direct_proto_preds, cls_direct_proto_scores = predict_with_prototypes(
            cls_X_test,
            cls_centroids,
            class_order
        )

        cls_direct_proto_results = evaluate_simple_classifier(
            y_test,
            cls_direct_proto_preds,
            cls_direct_proto_scores,
            class_order,
            name=f"CLS {language} direct (nearest Arabic weak centroid)"
        )



        # mean_centroids and cls_centroids should be your English weak-label training centroids
        # from the earlier training dataset.
        mean_R = fit_procrustes_from_centroids(
            mean_X_calib,
            y_calib,
            mean_centroids,
            class_order
        )

        cls_R = fit_procrustes_from_centroids(
            cls_X_calib,
            y_calib,
            cls_centroids,
            class_order
        )

        mean_X_calib_aligned = apply_rotation(mean_X_calib, mean_R)
        mean_X_test_aligned = apply_rotation(mean_X_test, mean_R)

        cls_X_calib_aligned = apply_rotation(cls_X_calib, cls_R)
        cls_X_test_aligned = apply_rotation(cls_X_test, cls_R)




        mean_ar_proc_direct = evaluate_probe_on_single_label(
            mean_X_test_aligned,
            mean_probe,
            y_test,
            class_order,
            name=f"Mean-pooled English probe on Procrustes-aligned {language}"
        )

        cls_ar_proc_direct = evaluate_probe_on_single_label(
            cls_X_test_aligned,
            cls_probe,
            y_test,
            class_order,
            name=f"CLS English probe on Procrustes-aligned {language}"

        )


        # 4.5 Geometry-only baselines on Procrustes-aligned {language}


        # Cosine to English weak probe weights
        mean_proc_weight_preds, mean_proc_weight_scores = predict_with_weight_cosine(
            mean_X_test_aligned,
            mean_probe,
            class_order
        )

        mean_proc_weight_results = evaluate_simple_classifier(
            y_test,
            mean_proc_weight_preds,
            mean_proc_weight_scores,
            class_order,
            name=f"Mean-pooled Procrustes-aligned {language} (cosine to English probe weights)"
        )

        cls_proc_weight_preds, cls_proc_weight_scores = predict_with_weight_cosine(
            cls_X_test_aligned,
            cls_probe,
            class_order
        )

        cls_proc_weight_results = evaluate_simple_classifier(
            y_test,
            cls_proc_weight_preds,
            cls_proc_weight_scores,
            class_order,
            name=f"CLS Procrustes-aligned {language} (cosine to English probe weights)"
        )


        # Nearest English weak-label centroid
        mean_proc_proto_preds, mean_proc_proto_scores = predict_with_prototypes(
            mean_X_test_aligned,
            mean_centroids,
            class_order
        )

        mean_proc_proto_results = evaluate_simple_classifier(
            y_test,
            mean_proc_proto_preds,
            mean_proc_proto_scores,
            class_order,
            name=f"Mean-pooled Procrustes-aligned {language} (nearest English weak centroid)"
        )

        cls_proc_proto_preds, cls_proc_proto_scores = predict_with_prototypes(
            cls_X_test_aligned,
            cls_centroids,
            class_order
        )

        cls_proc_proto_results = evaluate_simple_classifier(
            y_test,
            cls_proc_proto_preds,
            cls_proc_proto_scores,
            class_order,
            name=f"CLS Procrustes-aligned {language} (nearest English weak centroid)"
        )



        mean_ar_proc_bias = fit_bias_only(
            mean_X_calib_aligned,
            y_calib,
            mean_probe,
            class_order,

        )

        cls_ar_proc_bias = fit_bias_only(
            cls_X_calib_aligned,
            y_calib,
            cls_probe,
            class_order,

        )

        mean_ar_proc_bias_results = evaluate_fixed_weights_new_bias(
            mean_X_test_aligned,
            y_test,
            mean_probe,
            mean_ar_proc_bias,
            class_order,
            name=f"Mean-pooled Procrustes-aligned {language} + learned bias"
        )

        cls_ar_proc_bias_results = evaluate_fixed_weights_new_bias(
            cls_X_test_aligned,
            y_test,
            cls_probe,
            cls_ar_proc_bias,
            class_order,
            name=f"CLS Procrustes-aligned {language} + learned bias"
        )




        # Mean-pooled: {language} aligned embeddings -> English 6D emotion subspace
        mean_Z_calib_aligned = project_to_probe_subspace(
            mean_X_calib_aligned,
            mean_probe
        )

        mean_Z_test_aligned = project_to_probe_subspace(
            mean_X_test_aligned,
            mean_probe
        )

        mean_ar_proc_6d_results = evaluate_6d_logreg(
            mean_Z_calib_aligned,
            mean_Z_test_aligned,
            y_calib,
            y_test,
            class_order,
            name=f"Mean-pooled Procrustes-aligned {language}"
        )


        # CLS: {language} aligned embeddings -> English 6D emotion subspace
        cls_Z_calib_aligned = project_to_probe_subspace(
            cls_X_calib_aligned,
            cls_probe
        )

        cls_Z_test_aligned = project_to_probe_subspace(
            cls_X_test_aligned,
            cls_probe
        )

        cls_ar_proc_6d_results = evaluate_6d_logreg(
            cls_Z_calib_aligned,
            cls_Z_test_aligned,
            y_calib,
            y_test,
            class_order,
            name=f"CLS Procrustes-aligned {language}"
        )


    language_files = {
        "Chinese": ["xed_chinese.csv", "xed_tw.csv", "brighter_chinese.csv"],
        "Estonian": ["xed_estonian.csv"],
        "Russian": ["xed_russian.csv", "brighter_russian.csv"],
        "Swahili": ["brighter_swahili.csv"],
        "Ukrainian": ["brighter_ukranian.csv"],
        "German": ["xed_german.csv", "brighter_german.csv"],
    }
    for language, files in language_files.items():
        evaluate_language(language, files)
