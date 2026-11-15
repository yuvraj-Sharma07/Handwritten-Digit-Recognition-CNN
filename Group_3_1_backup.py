
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=14,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="digit_canvas",
            return_image_data=True,
        )

        # Read canvas image data
        canvas_data = getattr(canvas_result, "image_data", None)

        if canvas_data is not None:
            rgb = np.asarray(canvas_data)[..., :3]
            if rgb.size and np.max(rgb) > 0:
                input_img = Image.fromarray(
                    rgb.astype(np.uint8), mode="RGB"
                )

    else:
        uploaded = st.file_uploader(
            "Upload a digit image (PNG/JPG/JPEG)",
            type=["png", "jpg", "jpeg"],
        )
        if uploaded is not None:
            try:
                input_img = Image.open(uploaded).convert("RGB")
            except Exception:
                st.error("This image could not be opened. Please try another PNG or JPG.")

    if input_img is not None:
        st.image(input_img, caption="Your selected input", width=280)
    else:
        st.markdown(
            '<div class="hint">✏️ Draw a clear digit in the black canvas, '
            "or upload a picture. Your input preview will appear here.</div>",
            unsafe_allow_html=True,
        )

    st.caption(
        "Tip: use one large, centered digit with clear contrast. "
        "The model was trained on MNIST-style images."
    )

with result_col:
    st.subheader("2 · Prediction & explanation")

    if input_img is None:
        st.markdown(
            '<div class="hint">🔎 Your prediction, confidence, probability chart, '
            "and Grad-CAM explanation will appear here after you provide a digit.</div>",
            unsafe_allow_html=True,
        )
    else:
        processed = preprocess(input_img)
        tensor = tf.convert_to_tensor(processed[np.newaxis, ...], dtype=tf.float32)

        try:
            probabilities = np.asarray(model.predict(tensor, verbose=0)[0]).reshape(-1)
            if probabilities.size != 10:
                st.error(
                    f"Expected 10 class outputs (digits 0–9), but the model returned "
                    f"{probabilities.size}. Please check the trained model."
                )
                st.stop()

            predicted = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted]) * 100.0

            metric_col, note_col = st.columns([1, 1.5])
            with metric_col:
                st.metric("Predicted digit", str(predicted), f"{confidence:.1f}% confidence")
            with note_col:
                st.write("**Model input (28 × 28)**")
                st.image(processed.squeeze(), width=100, clamp=True)

            # Probability plot
            fig_prob, ax_prob = plt.subplots(figsize=(7, 2.8))
            ax_prob.bar(np.arange(10), probabilities)
            ax_prob.set_xticks(np.arange(10))
            ax_prob.set_xlabel("Digit")
            ax_prob.set_ylabel("Probability")
            ax_prob.set_ylim(0, max(1.0, float(np.max(probabilities)) * 1.12))
            ax_prob.set_title("Probability for each digit")
            ax_prob.grid(axis="y", alpha=0.2)
            fig_prob.tight_layout()
            st.pyplot(fig_prob)
            plt.close(fig_prob)

            heatmap = gradcam(model, tensor, predicted)
            if heatmap is not None:
                fig_cam, axes = plt.subplots(1, 3, figsize=(9, 3))
                axes[0].imshow(processed.squeeze(), cmap="gray", vmin=0, vmax=1)
                axes[0].set_title("Model input")
                axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
                axes[1].set_title("Grad-CAM")
                axes[2].imshow(processed.squeeze(), cmap="gray", vmin=0, vmax=1)
                axes[2].imshow(heatmap, cmap="jet", alpha=0.45, vmin=0, vmax=1)
                axes[2].set_title("Overlay")
                for ax in axes:
                    ax.axis("off")
                fig_cam.tight_layout()
                st.pyplot(fig_cam)
                plt.close(fig_cam)
                st.caption(
                    "Grad-CAM is a coarse visualization of influential regions. "
                    "It helps inspect a prediction but does not prove the prediction is correct."
                )
            else:
                st.info(
                    "The digit prediction worked, but Grad-CAM could not be generated "
                    "for this model architecture. The probability chart is still available."
                )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.caption(
                "Check that `models/digit_cnn.keras` is the model produced by "
                "`train_model.py` and that its input shape is 28 × 28 × 1."
            )

st.divider()
st.caption("Group 3 · Handwritten Digit Recognition with Explainable CNN")




