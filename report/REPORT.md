# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Phan Nguyễn Việt Nhân
**Nhóm:** Cá nhân
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Hai đoạn văn bản có nội dung hoặc ngữ cảnh tương đồng với nhau — các vector biểu diễn của chúng gần như cùng chỉ về một hướng trong không gian nhiều chiều, bất kể hai đoạn đó dài hay ngắn đến đâu.

**Ví dụ HIGH similarity:**
- Sentence A: "Quy định đánh giá luận văn thạc sĩ."
- Sentence B: "Các nguyên tắc khi bảo vệ đề án tốt nghiệp cao học."
- Tại sao tương đồng: Cả hai câu đều xoay quanh việc chấm điểm, nghiệm thu kết quả học tập ở bậc sau đại học — mô hình nhúng nhận ra chủ đề chung dù cách diễn đạt khác nhau.

**Ví dụ LOW similarity:**
- Sentence A: "Quy định đánh giá luận văn thạc sĩ."
- Sentence B: "Hướng dẫn làm bánh bông lan bơ tỏi."
- Tại sao khác: Hai câu thuộc hai lĩnh vực hoàn toàn không liên quan (giáo dục và ẩm thực), nên vector của chúng nằm ở hai hướng rất xa nhau trong không gian embedding.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity chỉ đo *góc* giữa hai vector chứ không quan tâm đến độ dài của chúng. Với văn bản, hai câu mang cùng ý nghĩa nhưng một câu dài, một câu ngắn sẽ có vector có magnitude rất khác nhau — dùng Euclidean distance lúc này sẽ bị ảnh hưởng bởi độ dài đó và cho kết quả sai. Cosine similarity loại bỏ vấn đề này nên phù hợp hơn để so sánh ngữ nghĩa.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> num_chunks = ceil((10,000 - 50) / (500 - 50)) = ceil(9,950 / 450) = **23 chunks**.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Với overlap = 100, bước nhảy giảm xuống còn 400 ký tự, nên num_chunks = ceil(9,900 / 400) = **25 chunks** (tăng thêm 2). Overlap lớn hơn đồng nghĩa với việc mỗi chunk "nhìn thấy" một phần của chunk kề bên — điều này giúp retriever không bị mất ngữ cảnh ở những câu nằm đúng chỗ ranh giới cắt, tránh trường hợp một quy định bị cắt đôi giữa chừng khiến câu trả lời bị thiếu thông tin.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** FAQ vin policy

**Tại sao nhóm chọn domain này?**
> Các văn bản quy chế đào tạo thường rất dài, ngôn từ mang tính pháp lý/học thuật cao và được chia thành cấu trúc phân cấp phức tạp (Chương, Điều, Khoản). Đây là một Use Case hoàn hảo cho hệ thống RAG vì sinh viên thường gặp khó khăn khi tra cứu thủ công, đồng thời nó đòi hỏi chiến lược phân mảnh dữ liệu (chunking) thông minh để không làm mất ngữ cảnh của từng điều luật.
### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | VU_HT02.VN Quy chế đào tạo Thạc sĩ | VinUniversity (ban hành 20/12/2022) | ~32,367 | chapter, doc_id, source |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| chapter | string | `quy_che_thac_si` | Lọc kết quả theo chủ đề quy chế, tránh trả về chunk không liên quan |
| doc_id | string | `VU_HT02.VN_Quy-che...` | Xác định chunk thuộc tài liệu gốc nào, hỗ trợ delete_document và truy vết nguồn |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| Quy chế Thạc sĩ | FixedSizeChunker (`fixed_size`) | 35 | 973 chars |  Cắt giữa câu, phá vỡ cấu trúc Điều/Khoản |
| Quy chế Thạc sĩ | SentenceChunker (`by_sentences`) | 73 | 441 chars |  Giữ trọn câu, nhưng chunk quá nhỏ |
| Quy chế Thạc sĩ | RecursiveChunker (`recursive`) | 37 | 873 chars |  Giữ nguyên paragraph theo cấu trúc Điều |

### Strategy Của Tôi

**Loại:** RecursiveChunker (chunk_size=1000)

**Mô tả cách hoạt động:**
> RecursiveChunker chia văn bản theo thứ tự ưu tiên các dấu tách: `\n\n` (paragraph) → `\n` (dòng) → `. ` (câu) → ` ` (từ). Nếu sau khi cắt một đoạn vẫn còn dài hơn chunk_size, nó tiếp tục đệ quy với dấu tách nhỏ hơn. Các mảnh nhỏ được gộp lại (merge) nếu tổng độ dài chưa vượt quá chunk_size, tránh lãng phí dung lượng chunk.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Văn bản quy chế VinUni được tổ chức theo cấu trúc phân cấp rõ ràng: Chương → Điều → Khoản, ngăn cách bởi dấu xuống dòng kép `\n\n`. RecursiveChunker khai thác pattern này bằng cách ưu tiên cắt theo paragraph, giữ nguyên từng Điều/Khoản trọn vẹn trong một chunk — phù hợp hơn so với fixed_size (cắt bừa) và sentence (chunk quá nhỏ).

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| Quy chế Thạc sĩ | fixed_size (baseline) | 35 | 973 | 3/5 queries đúng — hay cắt giữa Điều |
| Quy chế Thạc sĩ | **recursive (của tôi)** | 37 | 873 | 3/5 queries đúng — giữ Điều trọn vẹn hơn |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> RecursiveChunker là lựa chọn tốt nhất cho domain quy chế vì nó tôn trọng cấu trúc phân cấp tự nhiên của văn bản pháp lý. Khi mỗi chunk giữ được trọn vẹn 1 Điều hoặc 1 Khoản, hệ thống tìm kiếm dễ dàng trả về đúng đoạn văn bản chứa câu trả lời mà không bị mất bối cảnh.


## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Em dùng `re.split` với pattern lookbehind `(?<=[.!?])\s+|(?<=\.)\n` để tách câu tại các dấu `.` `!` `?` mà không làm mất dấu chấm câu đó. Sau khi có danh sách câu, em gom từng nhóm `max_sentences_per_chunk` câu lại thành một chunk bằng `" ".join(group)`.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Hàm `chunk` chỉ là wrapper gọi `_split` với toàn bộ danh sách separators. Trong `_split`, nếu đoạn văn đã nhỏ hơn `chunk_size` thì trả về luôn; nếu còn lớn hơn thì thử tách bằng separator ưu tiên nhất (`\n\n` trước, rồi `\n`, rồi `. `, ...). Các phần sau khi tách mà vẫn còn quá dài thì được đệ quy lại với separator tiếp theo trong danh sách.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> `add_documents` gọi `embedding_fn` cho từng document và lưu kết quả cùng metadata vào `_store` (list of dicts). Hàm `search` embed câu query, rồi duyệt qua toàn bộ `_store` để tính **tích vô hướng** (dot product) giữa query vector và từng document vector, sau đó sort giảm dần và trả về `top_k` kết quả đầu tiên.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter` lọc trước danh sách `_store` — chỉ giữ lại những record có metadata khớp toàn bộ key-value trong `metadata_filter` — rồi mới đưa tập con đó vào hàm `_search_records` để tính similarity. `delete_document` rebuild lại `_store` bằng list comprehension, loại bỏ mọi record có `doc_id` hoặc `id` trùng với đối số đầu vào, sau đó so sánh độ dài trước-sau để trả về `True/False`.

### KnowledgeBaseAgent

**`answer`** — approach:
> Em gọi `store.search(question, top_k=top_k)` để lấy các chunk liên quan nhất, nối nội dung chúng thành một đoạn `context`, rồi nhét vào prompt template cố định dạng: *"Use the following context to answer the question. Context: ... Question: ... Answer:"*. Cuối cùng đẩy prompt đó vào `llm_fn` và trả về chuỗi kết quả.

### Test Results

```
========================================================================================================== test session starts ===========================================================================================================
platform win32 -- Python 3.13.12, pytest-9.0.3, pluggy-1.6.0 -- C:\Users\phana\anaconda3\envs\AI_20K\python.exe
cachedir: .pytest_cache
rootdir: C:\GET_A_JOB\VIN_AI\Lab\2A202600279-PhanNguyenVietNhan-Day07
plugins: anyio-4.13.0
collected 42 items                                                                                                                                                                                                                        

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED                                                                                                                                               [  2%] 
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED                                                                                                                                                        [  4%] 
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED                                                                                                                                                 [  7%] 
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED                                                                                                                                                  [  9%] 
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED                                                                                                                                                       [ 11%] 
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED                                                                                                                                       [ 14%] 
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED                                                                                                                                             [ 16%] 
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED                                                                                                                                              [ 19%] 
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED                                                                                                                                            [ 21%] 
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED                                                                                                                                                              [ 23%] 
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED                                                                                                                                              [ 26%] 
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED                                                                                                                                                         [ 28%] 
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED                                                                                                                                                     [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED                                                                                                                                                               [ 33%] 
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED                                                                                                                                      [ 35%] 
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED                                                                                                                                          [ 38%] 
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED                                                                                                                                    [ 40%] 
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED                                                                                                                                          [ 42%] 
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED                                                                                                                                                              [ 45%] 
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED                                                                                                                                                [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED                                                                                                                                                  [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED                                                                                                                                                        [ 52%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED                                                                                                                                             [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED                                                                                                                                               [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED                                                                                                                                   [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED                                                                                                                                                [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                                                                                                                                                         [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED                                                                                                                                                        [ 66%] 
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED                                                                                                                                                   [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED                                                                                                                                               [ 71%] 
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED                                                                                                                                          [ 73%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED                                                                                                                            [ 90%] 
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED                                                                                                                                       [ 92%] 
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED                                                                                                                                [ 95%] 
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED                                                                                                                      [ 97%] 
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED                                                                                                                          [100%] 

=========================================================================================================== 42 passed in 1.30s ===========================================================================================================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "Quy định điểm chuẩn đạt tốt nghiệp" | "Điểm số để được cấp bằng ra trường" | high | > 0.85 | Đúng |
| 2 | "Đình chỉ học tập do gian lận thi cử" | "Cách chọn món tráng miệng ngon" | low | < 0.1 | Đúng |
| 3 | "Luận văn thạc sĩ cần 15 tín chỉ" | "Học viên cần 15 tín chỉ cho luận văn môn học" | high | > 0.90 | Đúng |
| 4 | "Xin nghỉ thai sản tạm thời" | "Nghỉ học do ốm đau bệnh tật" | high | > 0.70 | Đúng |
| 5 | "AI tạo sinh và Large Language Models" | "Luật xử phạt gian lận trong RAG" | low | ~ 0.20 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Cặp 4 ("xin nghỉ thai sản" vs "nghỉ học do ốm đau") cho điểm trên 0.70 — khá bất ngờ vì hai câu dùng từ vựng khác nhau hoàn toàn nhưng mô hình vẫn nhận ra chúng cùng thuộc ngữ cảnh "tạm ngừng học tập vì lý do cá nhân". Điều này cho thấy embedding không đơn thuần đếm từ giống nhau mà thực sự nắm bắt được *ý nghĩa ngầm* của câu trong không gian nhiều chiều — ví dụ như "học viên" và "sinh viên" được coi gần như đồng nghĩa dù hai từ hoàn toàn khác nhau về mặt ký tự.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Thời gian tối thiểu thực hiện luận văn thạc sĩ là bao lâu? | Ít nhất 06 tháng |
| 2 | Điểm luận văn bao nhiêu thì được xếp loại đạt? | Lớn hơn hoặc bằng 5,5 điểm |
| 3 | Học viên thi hộ hoặc nhờ thi hộ bị xử lý kỷ luật thế nào? | Đình chỉ 1 năm lần đầu, buộc thôi học lần 2 |
| 4 | Số tín chỉ được công nhận và chuyển đổi tối đa là bao nhiêu? | Không vượt quá 30 tín chỉ |
| 5 | Hội đồng đánh giá luận văn cần có ít nhất bao nhiêu thành viên? | Ít nhất 05 thành viên |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Thời gian luận văn | "Học viên thực hiện luận văn trong thời gian ít nhất 06 tháng" | 0.4829 | Có | Context chứa đúng thông tin 06 tháng |
| 2 | Điểm đạt luận văn | "Người hướng dẫn không được cho điểm đánh giá" (nhầm!) | 0.4896 | Không | Top-1 sai, chunk chứa "5,5 điểm" nằm ở top-3 |
| 3 | Thi hộ kỷ luật | "Thi hộ đều bị kỷ luật đình chỉ 01 năm...buộc thôi học" | 0.7110 | Có | Trả về chính xác điều khoản kỷ luật |
| 4 | Tín chỉ chuyển đổi tối đa | "Số tín chỉ được công nhận không vượt quá 15 tín chỉ" | 0.5939 | ~~ | Top-1 ghi 15 TC (Khoản 2), gold answer 30 TC (Khoản 1) |
| 5 | Hội đồng bao nhiêu người | "Người hướng dẫn tham gia hội đồng với tư cách ủy viên" | 0.5677 | ~~ | Nói về vai trò hội đồng, không ghi rõ con số 05 |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 3 / 5

**Nhận xét:**
> Query 2 (điểm đạt luận văn) bị nhiễu vì chunk top-1 nói về người hướng dẫn không được cho điểm — cùng ngữ cảnh "điểm" nhưng sai Điều. Query 4 và 5 cho thấy RecursiveChunker đôi khi tách một điều khoản có nhiều khoản con thành nhiều chunk riêng biệt, khiến chunk chứa con số cụ thể (30 TC, 05 thành viên) không phải lúc nào cũng leo lên top-1.

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Bạn trong nhóm chỉ ra rằng dùng lookbehind regex `(?<=[.!?])` thay vì `str.split('.')` giải quyết được vấn đề mất dấu chấm câu — cái mà em lúc đầu không chú ý. Sau khi xem lại code thì thấy cách này sạch hơn và ít edge case hơn rất nhiều.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Một nhóm khác dùng thêm trường `date_issued` trong metadata để lọc bỏ các văn bản cũ, tránh trường hợp AI trả về quy định đã hết hiệu lực. Đây là use case thực tế mà em không nghĩ tới — metadata không chỉ dùng để xác định nguồn mà còn dùng để kiểm soát *tính thời sự* của thông tin.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Em sẽ thêm bước trích xuất metadata có cấu trúc trước khi đưa tài liệu vào embedding — ví dụ dùng regex hoặc một mô hình NER nhỏ để nhận diện số Điều, số Khoản rồi gán thành metadata `dieu_khoan`. Khi đó có thể kết hợp BM25 Hybrid Search để tìm chính xác theo số điều khoản, thay vì chỉ phụ thuộc vào similarity vector như hiện tại.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 0 / 5 |
| **Tổng** | | **95 / 100** |
