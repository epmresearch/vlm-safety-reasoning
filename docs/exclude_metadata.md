1. The Inference-Time Reality
If your end goal is to build an autonomous pipeline that can ingest a raw camera feed from a construction site and output safety rule violations and bounding boxes, you will not have a human in the loop to pre-label the lighting or camera distance at inference time. If your model learns to rely on those text hints in the prompt during fine-tuning, its performance will degrade when those hints are suddenly missing in the real world.

2. Information Leakage and Redundancy
The visual tokens extracted from the image already contain all the information regarding whether a scene is "underexposed" or a "long distance" shot. Feeding this into the text prompt is essentially spoon-feeding the model redundant features. Forcing the model's visual encoder and cross-attention layers to deduce the spatial hierarchy and lighting conditions directly from the pixels will yield a much more robust representation of the scene.

3. How to Actually Use This Metadata
Even though you are stripping these labels from the prompt, you should not delete them from your data loader. You will very likely need them for the research paper.

When it comes time to report your findings, these labels are highly valuable for stratified evaluation. You will want to run your test set and group your performance metrics (like bounding box IoU or exact match for rule violations) by these attributes.

This allows you to write compelling analytical sections in your paper, such as:

“The model maintains 85% accuracy on 'normal lighting' but drops to 62% on 'underexposed' images.”

“Bounding box precision for the 'excavator' class degraded significantly in 'long distance' views.”






Based on the paper provided, no, they did not include the metadata (quality of information, illumination, camera distance, and view) in the prompts when evaluating the different models.

Instead of feeding these conditions into the prompt, they assigned these metadata labels as tags or attributes to the images within their dataset itself. Their goal was to provide these labels so that future researchers could filter the dataset to create specific or challenging training subsets, rather than using them to guide the vision-language models during inference.



Stratified Sampling: When setting up your PyTorch data loaders with Khashayar, use the metadata to ensure your training batches are perfectly balanced across edge cases (e.g., guaranteeing an equal split of nighttime vs. daytime shots, or long-distance vs. short-distance).


you can use these metadata tags to slice your error analysis. If the model struggles specifically with "underexposed" images, you can generate preference pairs (chosen/rejected outputs) specifically for those low-visibility conditions and optimize using DPO or PPO to penalize those specific types of hallucinations.



1. For Curriculum Learning and Stratified Sampling (Recommended)
You do not need to include the metadata in the text prompt itself. Instead, use those tags to structure your training loops.

Stratified Sampling: When setting up your PyTorch data loaders with Khashayar, use the metadata to ensure your training batches are perfectly balanced across edge cases (e.g., guaranteeing an equal split of nighttime vs. daytime shots, or long-distance vs. short-distance).

Curriculum Learning: You can build a curriculum that trains the VLM on "easy" images first (short distance, normal lighting, sparse info) and gradually introduces the harder, noisier images (long distance, underexposed, rich info) as the loss stabilizes.

2. As a Prompt Conditioning Signal
If you do decide to include the metadata in the prompt during fine-tuning (e.g., <System>: You are an inspector. [Metadata: Illumination=Night, Distance=Long]), you are teaching the model's cross-attention layers to condition their visual feature extraction on those specific textual priors.

The Catch: If you fine-tune the model to expect these metadata tags in the prompt, you must also provide them during inference. If a user later uploads an image without those tags, the model will experience a distribution shift and performance will degrade.

3. For Alignment and Preference Learning (DPO/RLHF)
Given your focus on model alignment and preference learning algorithms, this metadata becomes incredibly valuable post-supervised fine-tuning.

Models frequently hallucinate safety violations in poor lighting or long-distance shots.

When evaluating the model with Dr. Zangeneh, you can use these metadata tags to slice your error analysis. If the model struggles specifically with "underexposed" images, you can generate preference pairs (chosen/rejected outputs) specifically for those low-visibility conditions and optimize using DPO or PPO to penalize those specific types of hallucinations.


The takeaway: If you want a robust, general-purpose inspector model, keep the fine-tuning prompts clean and realistic (no metadata in the prompt) just like the paper did. Use the metadata purely on the backend to balance your datasets, structure your training loops, and guide your alignment phase.

