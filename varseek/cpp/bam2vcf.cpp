// bam2vcf -- stream a coordinate-sorted BAM/CRAM and emit a sites-only VCF of every
// variant recorded in the alignments themselves (CIGAR I/D operations plus mismatches
// taken from the MD tag or from a reference FASTA).
//
// This is the fast path behind varseek.utils.bam_to_vcf(); it is a drop-in replacement
// for the Python `parse_cigars` walk, which is bounded by the per-read CPython loop.
//
// Design notes
// ------------
//   * One pass. Reference bases for the read's aligned span are materialised into a
//     small reusable buffer -- from the FASTA when one is given, otherwise reconstructed
//     from the MD tag -- and then a single CIGAR walk reads mismatches and indels out of
//     it. Both input modes therefore share one code path.
//   * Depth (DP) is accumulated in the same pass using a ring buffer of per-position
//     slots. Because the input is coordinate sorted, no read starting at S can carry a
//     variant at, or contribute coverage to, any position < S -- so once a read at S is
//     reached, every position below S is final and can be emitted and evicted. Memory is
//     therefore O(longest read reference span), not O(genome).
//   * Variants at a position live in a short vector attached to that position's slot, so
//     aggregation is a linear scan over 0-3 entries rather than a hash lookup.
//   * Emission is in ascending coordinate order (contigs follow BAM header order), so the
//     output is tabix-indexable without a sort.
//
// Build: see varseek/utils/native.py, which compiles and caches this on demand.

#include <cctype>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include <htslib/bgzf.h>
#include <htslib/faidx.h>
#include <htslib/hts.h>
#include <htslib/regidx.h>
#include <htslib/sam.h>
#include <htslib/tbx.h>
#include <htslib/thread_pool.h>

#define BAM2VCF_VERSION "1.0.0"

namespace {

// ---------------------------------------------------------------------------
// Output: plain text or BGZF, buffered.
// ---------------------------------------------------------------------------
class Writer {
  public:
    bool open(const std::string &path, bool bgzip, htsThreadPool *pool) {
        bgzip_ = bgzip;
        if (path == "-" || path.empty()) {
            if (bgzip_) {
                bgz_ = bgzf_open("-", "w");
                if (!bgz_) return false;
            } else {
                plain_ = stdout;
                owns_plain_ = false;
            }
        } else if (bgzip_) {
            bgz_ = bgzf_open(path.c_str(), "w");
            if (!bgz_) return false;
        } else {
            plain_ = fopen(path.c_str(), "w");
            if (!plain_) return false;
            owns_plain_ = true;
        }
        if (bgz_ && pool && pool->pool) bgzf_thread_pool(bgz_, pool->pool, 0);
        buf_.reserve(kFlushAt + 4096);
        return true;
    }

    void put(const char *data, size_t len) {
        buf_.append(data, len);
        if (buf_.size() >= kFlushAt) flush();
    }
    void put(const std::string &s) { put(s.data(), s.size()); }

    bool flush() {
        if (buf_.empty()) return true;
        bool ok = true;
        if (bgz_) {
            ok = bgzf_write(bgz_, buf_.data(), buf_.size()) == (ssize_t)buf_.size();
        } else if (plain_) {
            ok = fwrite(buf_.data(), 1, buf_.size(), plain_) == buf_.size();
        }
        buf_.clear();
        if (!ok) failed_ = true;
        return ok;
    }

    bool close() {
        flush();
        if (bgz_) {
            if (bgzf_close(bgz_) < 0) failed_ = true;
            bgz_ = nullptr;
        }
        if (plain_) {
            if (owns_plain_ && fclose(plain_) != 0) failed_ = true;
            plain_ = nullptr;
        }
        return !failed_;
    }

    bool failed() const { return failed_; }

  private:
    static const size_t kFlushAt = 1u << 20;
    FILE *plain_ = nullptr;
    BGZF *bgz_ = nullptr;
    bool owns_plain_ = false;
    bool bgzip_ = false;
    bool failed_ = false;
    std::string buf_;
};

// ---------------------------------------------------------------------------
// Reference FASTA access, cached in chunks.
//
// Reads arrive in ascending coordinate order, so a single sliding chunk keeps the
// hit rate near 1. The left margin exists so that left-aligning an indel can look
// backwards without forcing a refetch.
// ---------------------------------------------------------------------------
class RefCache {
  public:
    explicit RefCache(faidx_t *fai) : fai_(fai) {}

    bool ok() const { return fai_ != nullptr; }

    // Pointer to reference bases for 1-based inclusive [beg, beg+len-1], or nullptr if
    // unavailable (contig absent from the FASTA, or the span runs off its end).
    const char *span(const char *chrom, int64_t beg, int64_t len) {
        if (!fai_ || len <= 0) return nullptr;
        if (!load(chrom, beg, beg + len - 1)) return nullptr;
        return seq_.data() + (beg - start_);
    }

    // Single 1-based base, or 0 if unavailable.
    char base(const char *chrom, int64_t pos) {
        const char *p = span(chrom, pos, 1);
        return p ? *p : 0;
    }

    bool contig_missing() const { return missing_; }

  private:
    static const int64_t kChunk = 1 << 20;   // 1 Mb fetches
    static const int64_t kMargin = 1 << 16;  // keep 64 kb behind for left-alignment

    bool load(const char *chrom, int64_t beg, int64_t end) {
        if (chrom_ == chrom && beg >= start_ && end <= end_) return true;
        if (last_failed_chrom_ == chrom && beg >= failed_beg_) {
            // Same contig already known to be absent/short; don't hammer faidx.
            if (failed_hard_) return false;
        }
        int64_t want_start = beg > kMargin ? beg - kMargin : 1;
        int64_t want_end = want_start + kChunk - 1;
        if (want_end < end) want_end = end;

        hts_pos_t got = 0;
        char *s = faidx_fetch_seq64(fai_, chrom, want_start - 1, want_end - 1, &got);
        if (!s || got <= 0) {
            if (s) free(s);
            last_failed_chrom_ = chrom;
            failed_beg_ = beg;
            failed_hard_ = (faidx_has_seq(fai_, chrom) == 0);
            if (failed_hard_) missing_ = true;
            return false;
        }
        seq_.assign(s, (size_t)got);
        free(s);
        for (char &c : seq_) c = (char)toupper((unsigned char)c);
        chrom_ = chrom;
        start_ = want_start;
        end_ = want_start + (int64_t)seq_.size() - 1;
        return beg >= start_ && end <= end_;
    }

    faidx_t *fai_ = nullptr;
    std::string seq_;
    std::string chrom_;
    int64_t start_ = 1, end_ = 0;
    std::string last_failed_chrom_;
    int64_t failed_beg_ = 0;
    bool failed_hard_ = false;
    bool missing_ = false;
};

// ---------------------------------------------------------------------------
// MD tag reader.
//
// MD describes only the reference bases covered by M/=/X and D operations: it says
// nothing about soft clips, insertions or spliced gaps, so it cannot be walked on its
// own to derive coordinates. It is consumed here in lockstep with the CIGAR.
// ---------------------------------------------------------------------------
class MDReader {
  public:
    explicit MDReader(const char *md) : p_(md) {}

    // Next reference base for an M/=/X position. Returns 0 for "matches the read",
    // the reference base for a mismatch, or -1 on malformed input.
    int next_aligned() {
        if (run_ > 0) {
            --run_;
            return 0;
        }
        while (*p_) {
            if (isdigit((unsigned char)*p_)) {
                int64_t n = 0;
                while (isdigit((unsigned char)*p_)) n = n * 10 + (*p_++ - '0');
                if (n == 0) continue;  // MD emits 0-length runs between adjacent mismatches
                run_ = n - 1;
                return 0;
            }
            if (*p_ == '^') break;  // a deletion token where an aligned base was expected
            if (isalpha((unsigned char)*p_)) return toupper((unsigned char)*p_++);
            ++p_;  // tolerate stray characters
        }
        bad_ = true;
        return -1;
    }

    // Deleted reference bases for a CIGAR D of the given length.
    bool read_deletion(int len, char *out) {
        if (run_ != 0) {
            bad_ = true;
            return false;
        }
        while (isdigit((unsigned char)*p_)) {
            int64_t n = 0;
            while (isdigit((unsigned char)*p_)) n = n * 10 + (*p_++ - '0');
            if (n != 0) {  // a real match run cannot sit between the CIGAR and its ^ token
                bad_ = true;
                return false;
            }
        }
        if (*p_ != '^') {
            bad_ = true;
            return false;
        }
        ++p_;
        for (int i = 0; i < len; ++i) {
            if (!isalpha((unsigned char)*p_)) {
                bad_ = true;
                return false;
            }
            out[i] = (char)toupper((unsigned char)*p_++);
        }
        return true;
    }

    bool bad() const { return bad_; }

  private:
    const char *p_;
    int64_t run_ = 0;
    bool bad_ = false;
};

// ---------------------------------------------------------------------------
// Per-position accumulator.
// ---------------------------------------------------------------------------
enum VarType { kSNV = 0, kIns = 1, kDel = 2 };

struct Allele {
    std::string ref;
    std::string alt;
    uint32_t ao = 0;
    uint8_t type = kSNV;
};

struct Slot {
    uint32_t depth = 0;
    std::vector<Allele> alleles;
};

struct Config {
    std::string bam;
    std::string reference;
    std::string out = "-";
    std::string regions;
    std::string stats_json;
    int threads = 1;
    uint32_t min_count = 1;
    double min_vaf = 0.0;  // inclusive bounds; the pair is inactive at 0.0 / 1.0
    double max_vaf = 1.0;
    int min_mapq = 0;
    int min_baseq = 0;
    uint16_t exclude_flags = BAM_FUNMAP | BAM_FSECONDARY | BAM_FQCFAIL | BAM_FDUP | BAM_FSUPPLEMENTARY;
    uint16_t require_flags = 0;
    int64_t max_reads = 0;  // 0 = no limit
    int64_t flush_margin = 1000;
    bool normalize = true;
    bool skip_indels = false;
    bool bgzip = false;
    bool index = false;
    bool assume_sorted = false;
    bool strip_version = false;
    bool progress = false;
    bool emit_type = true;
};

struct Stats {
    int64_t reads_total = 0;
    int64_t reads_skipped_flag = 0;
    int64_t reads_skipped_mapq = 0;
    int64_t reads_skipped_region = 0;
    int64_t reads_used = 0;
    int64_t reads_fast_path = 0;   // NM==0 and no indels: depth only
    int64_t reads_with_md = 0;
    int64_t reads_bad_md = 0;
    int64_t reads_no_ref = 0;
    int64_t reads_no_ref_bases = 0;  // neither MD nor FASTA usable
    int64_t sites_emitted = 0;
    int64_t records_emitted = 0;
    int64_t alleles_seen = 0;
    int64_t dropped_min_count = 0;
    int64_t dropped_vaf = 0;
    int64_t dropped_region = 0;
    int64_t skipped_no_anchor = 0;
    int64_t shifts_clamped = 0;
};

class Caller {
  public:
    Caller(const Config &cfg, sam_hdr_t *hdr, RefCache &ref, regidx_t *reg, regitr_t *regitr, Writer &out)
        : cfg_(cfg), hdr_(hdr), ref_(ref), reg_(reg), regitr_(regitr), out_(out) {
        ring_.resize(kInitialCap);
        cap_ = kInitialCap;
        mask_ = cap_ - 1;
        margin_ = cfg_.normalize ? cfg_.flush_margin : 0;
        vaf_filter_ = cfg_.min_vaf > 0.0 || cfg_.max_vaf < 1.0;
    }

    Stats &stats() { return stats_; }

    // Called once per accepted read, in ascending coordinate order.
    void add_read(bam1_t *b) {
        const int32_t tid = b->core.tid;
        const int64_t start1 = (int64_t)b->core.pos + 1;

        if (tid != tid_) {
            flush_all();
            tid_ = tid;
            const char *name = sam_hdr_tid2name(hdr_, tid);
            chrom_ = name ? name : ".";
            out_chrom_ = display_name(chrom_.c_str());
            win_start_ = start1;
            win_end_ = start1 - 1;
            head_ = 0;
        } else {
            int64_t frontier = start1 - margin_;
            if (frontier > win_start_) flush_upto(frontier);
        }

        walk(b, start1);
    }

    void finish() { flush_all(); }

    void write_header() {
        char vaf_buf[64];
        snprintf(vaf_buf, sizeof(vaf_buf), ",minVAF=%g,maxVAF=%g", cfg_.min_vaf, cfg_.max_vaf);
        std::string h;
        h += "##fileformat=VCFv4.2\n";
        h += "##source=varseek-bam2vcf-" BAM2VCF_VERSION "\n";
        h += "##bam2vcfCommandLine=<bam=\"" + cfg_.bam + "\",minCount=" + std::to_string(cfg_.min_count) +
             vaf_buf + ",minMapQ=" + std::to_string(cfg_.min_mapq) + ",minBaseQ=" + std::to_string(cfg_.min_baseq) +
             ",normalize=" + (cfg_.normalize ? "1" : "0") + ">\n";
        if (!cfg_.reference.empty()) h += "##reference=file://" + cfg_.reference + "\n";
        const int nref = sam_hdr_nref(hdr_);
        for (int i = 0; i < nref; ++i) {
            const char *name = sam_hdr_tid2name(hdr_, i);
            const int64_t len = sam_hdr_tid2len(hdr_, i);
            if (!name) continue;
            h += "##contig=<ID=" + display_name(name) + ",length=" + std::to_string(len) + ">\n";
        }
        h += "##INFO=<ID=AO,Number=A,Type=Integer,Description=\"Reads supporting the alternate allele\">\n";
        h += "##INFO=<ID=DP,Number=1,Type=Integer,Description=\"Reads passing filters that span this position\">\n";
        h += "##INFO=<ID=VAF,Number=A,Type=Float,Description=\"Alternate allele fraction, AO/DP\">\n";
        if (cfg_.emit_type) h += "##INFO=<ID=TYPE,Number=A,Type=String,Description=\"Variant class: snv, ins or del\">\n";
        h += "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n";
        out_.put(h);
    }

  private:
    static const size_t kInitialCap = 1u << 12;

    std::string display_name(const char *name) const {
        std::string s(name ? name : ".");
        if (cfg_.strip_version) {
            size_t dot = s.rfind('.');
            if (dot != std::string::npos && dot + 1 < s.size()) {
                bool all_digits = true;
                for (size_t i = dot + 1; i < s.size(); ++i)
                    if (!isdigit((unsigned char)s[i])) all_digits = false;
                if (all_digits) s.resize(dot);
            }
        }
        return s;
    }

    // --- ring buffer -------------------------------------------------------
    size_t idx(int64_t pos) const { return (head_ + (size_t)(pos - win_start_)) & mask_; }

    void grow(size_t need) {
        size_t ncap = cap_;
        while (ncap < need) ncap <<= 1;
        std::vector<Slot> nring(ncap);
        int64_t span = win_end_ >= win_start_ ? win_end_ - win_start_ + 1 : 0;
        for (int64_t i = 0; i < span; ++i) nring[(size_t)i] = std::move(ring_[idx(win_start_ + i)]);
        ring_.swap(nring);
        cap_ = ncap;
        mask_ = cap_ - 1;
        head_ = 0;
    }

    void reserve_upto(int64_t pos) {
        if (pos < win_start_) return;
        size_t need = (size_t)(pos - win_start_ + 1);
        if (need > cap_) grow(need);
        if (pos > win_end_) win_end_ = pos;
    }

    void add_depth(int64_t pos1, int64_t len) {
        if (len <= 0) return;
        reserve_upto(pos1 + len - 1);
        for (int64_t p = pos1; p < pos1 + len; ++p) {
            if (p < win_start_) continue;
            ring_[idx(p)].depth++;
        }
    }

    void add_allele(int64_t pos1, const char *ref, size_t rlen, const char *alt, size_t alen, uint8_t type) {
        if (pos1 < win_start_) {  // only reachable if a clamp failed; keeps output sorted
            stats_.shifts_clamped++;
            return;
        }
        reserve_upto(pos1);
        std::vector<Allele> &v = ring_[idx(pos1)].alleles;
        for (Allele &a : v) {
            if (a.type == type && a.ref.size() == rlen && a.alt.size() == alen &&
                memcmp(a.ref.data(), ref, rlen) == 0 && memcmp(a.alt.data(), alt, alen) == 0) {
                a.ao++;
                return;
            }
        }
        Allele a;
        a.ref.assign(ref, rlen);
        a.alt.assign(alt, alen);
        a.ao = 1;
        a.type = type;
        v.push_back(std::move(a));
        stats_.alleles_seen++;
    }

    void flush_upto(int64_t upto) {
        if (upto <= win_start_) return;
        int64_t last = upto - 1 < win_end_ ? upto - 1 : win_end_;
        for (int64_t p = win_start_; p <= last; ++p) {
            Slot &s = ring_[idx(p)];
            if (!s.alleles.empty()) emit(p, s);
            s.depth = 0;
            if (s.alleles.capacity() > 8)
                std::vector<Allele>().swap(s.alleles);
            else
                s.alleles.clear();
            head_ = (head_ + 1) & mask_;
            win_start_ = p + 1;
        }
        if (upto > win_end_) {
            win_start_ = upto;
            win_end_ = upto - 1;
            head_ = 0;
        }
    }

    void flush_all() {
        if (tid_ < 0) return;
        flush_upto(win_end_ + 1);
    }

    // --- emission ---------------------------------------------------------
    void emit(int64_t pos1, Slot &s) {
        bool wrote_any = false;
        for (const Allele &a : s.alleles) {
            if (a.ao < cfg_.min_count) {
                stats_.dropped_min_count++;
                continue;
            }
            // VAF is undefined without depth, so a site with DP==0 cannot satisfy a VAF
            // bound and is dropped rather than silently passed through.
            const double vaf = s.depth > 0 ? (double)a.ao / (double)s.depth : -1.0;
            if (vaf_filter_ && (s.depth == 0 || vaf < cfg_.min_vaf || vaf > cfg_.max_vaf)) {
                stats_.dropped_vaf++;
                continue;
            }
            if (reg_) {
                // regidx uses 0-based inclusive coordinates.
                int64_t beg0 = pos1 - 1;
                int64_t end0 = beg0 + (int64_t)a.ref.size() - 1;
                if (!regidx_overlap(reg_, chrom_.c_str(), beg0, end0, regitr_)) {
                    stats_.dropped_region++;
                    continue;
                }
            }
            char head[512];
            int n = snprintf(head, sizeof(head), "%s\t%" PRId64 "\t.\t", out_chrom_.c_str(), pos1);
            line_.assign(head, (size_t)n);
            line_ += a.ref;
            line_ += '\t';
            line_ += a.alt;
            line_ += "\t.\t.\tAO=";
            line_ += std::to_string(a.ao);
            line_ += ";DP=";
            line_ += std::to_string(s.depth);
            line_ += ";VAF=";
            if (s.depth > 0) {
                char vaf_buf[32];
                int m = snprintf(vaf_buf, sizeof(vaf_buf), "%.6g", vaf);
                line_.append(vaf_buf, (size_t)m);
            } else {
                line_ += '.';
            }
            if (cfg_.emit_type) {
                line_ += ";TYPE=";
                line_ += (a.type == kSNV ? "snv" : (a.type == kIns ? "ins" : "del"));
            }
            line_ += '\n';
            out_.put(line_);
            stats_.records_emitted++;
            wrote_any = true;
        }
        if (wrote_any) stats_.sites_emitted++;
    }

    // --- indel left-alignment ---------------------------------------------
    // Rotates `s` left while the preceding reference base equals its last base, which is
    // the standard left-shift for an indel inside a repeat (what `bcftools norm` does).
    // `first1` is the 1-based position of the first base of `s` -- for an insertion, the
    // position the inserted bases would occupy. The shift is clamped so the resulting
    // anchor (first1 - 1) never falls behind the flush frontier, which keeps the output
    // coordinate sorted.
    void left_shift(std::string &s, int64_t &first1) {
        if (!cfg_.normalize || !ref_.ok() || s.empty()) return;
        int64_t budget = cfg_.flush_margin;
        while (budget-- > 0 && first1 > 1 && (first1 - 1) > win_start_) {
            char prev = ref_.base(chrom_.c_str(), first1 - 1);
            if (!prev || prev != s.back()) break;
            s.insert(s.begin(), prev);
            s.pop_back();
            --first1;
        }
    }

    char anchor_base(int64_t anchor1, const char *refspan, int64_t span_start1, int64_t span_len) {
        if (ref_.ok()) {
            char c = ref_.base(chrom_.c_str(), anchor1);
            if (c) return c;
        }
        int64_t off = anchor1 - span_start1;
        if (refspan && off >= 0 && off < span_len) return refspan[off];
        return 0;
    }

    // --- the per-read walk ------------------------------------------------
    void walk(bam1_t *b, int64_t start1) {
        const uint32_t *cig = bam_get_cigar(b);
        const int ncig = b->core.n_cigar;
        const uint8_t *qseq = bam_get_seq(b);
        const uint8_t *qual = bam_get_qual(b);

        int64_t span_len = 0;
        bool has_indel = false;
        for (int i = 0; i < ncig; ++i) {
            const int op = bam_cigar_op(cig[i]);
            const int len = bam_cigar_oplen(cig[i]);
            if (bam_cigar_type(op) & 2) span_len += len;  // consumes reference
            if (op == BAM_CINS || op == BAM_CDEL) has_indel = true;
        }
        if (span_len <= 0) return;

        // Fast path: NM==0 with no indels means the read matches the reference exactly,
        // so it contributes depth only and neither the MD tag nor the FASTA is needed.
        bool depth_only = false;
        if (!has_indel) {
            const uint8_t *nm = bam_aux_get(b, "NM");
            if (nm) {
                int64_t v = bam_aux2i(nm);
                if (v == 0) depth_only = true;
            }
        }

        const char *refspan = nullptr;
        if (!depth_only) {
            refspan = build_refspan(b, start1, span_len, cig, ncig, qseq);
            if (!refspan) depth_only = true;  // no usable reference bases for this read
        } else {
            stats_.reads_fast_path++;
        }

        int64_t rpos1 = start1;  // 1-based reference position of the next aligned base
        int qpos = 0;

        for (int i = 0; i < ncig; ++i) {
            const int op = bam_cigar_op(cig[i]);
            const int len = bam_cigar_oplen(cig[i]);
            switch (op) {
                case BAM_CMATCH:
                case BAM_CEQUAL:
                case BAM_CDIFF: {
                    add_depth(rpos1, len);
                    if (!depth_only) {
                        const int64_t base_off = rpos1 - start1;
                        for (int k = 0; k < len; ++k) {
                            const char rb = refspan[base_off + k];
                            if (!rb || rb == 'N') continue;
                            const char qb = seq_nt16_str[bam_seqi(qseq, qpos + k)];
                            if (qb == rb || qb == 'N' || qb == '=') continue;
                            if (cfg_.min_baseq > 0 && qual[qpos + k] < cfg_.min_baseq) continue;
                            add_allele(rpos1 + k, &rb, 1, &qb, 1, kSNV);
                        }
                    }
                    rpos1 += len;
                    qpos += len;
                    break;
                }
                case BAM_CINS: {
                    if (!depth_only && !cfg_.skip_indels)
                        emit_insertion(b, qseq, qual, qpos, len, rpos1, refspan, start1, span_len);
                    qpos += len;
                    break;
                }
                case BAM_CDEL: {
                    add_depth(rpos1, len);  // the read spans deleted bases
                    if (!depth_only && !cfg_.skip_indels) emit_deletion(len, rpos1, refspan, start1, span_len);
                    rpos1 += len;
                    break;
                }
                case BAM_CREF_SKIP:  // spliced gap: reference skip, not a deletion, and absent from MD
                    rpos1 += len;
                    break;
                case BAM_CSOFT_CLIP:  // present in the query sequence, so it advances qpos
                    qpos += len;
                    break;
                default:  // H and P consume neither the read nor the reference
                    break;
            }
        }
    }

    void emit_insertion(bam1_t *b, const uint8_t *qseq, const uint8_t *qual, int qpos, int len, int64_t rpos1,
                        const char *refspan, int64_t span_start1, int64_t span_len) {
        (void)b;
        if (cfg_.min_baseq > 0) {
            for (int k = 0; k < len; ++k)
                if (qual[qpos + k] < cfg_.min_baseq) return;
        }
        ins_.clear();
        ins_.reserve((size_t)len);
        for (int k = 0; k < len; ++k) ins_ += seq_nt16_str[bam_seqi(qseq, qpos + k)];

        int64_t first1 = rpos1;  // the inserted bases would sit at, and after, rpos1
        left_shift(ins_, first1);
        const int64_t anchor1 = first1 - 1;
        if (anchor1 < 1) {  // nothing to anchor on at the very start of the contig
            stats_.skipped_no_anchor++;
            return;
        }
        const char anchor = anchor_base(anchor1, refspan, span_start1, span_len);
        if (!anchor) {
            stats_.skipped_no_anchor++;
            return;
        }
        alt_.assign(1, anchor);
        alt_ += ins_;
        add_allele(anchor1, &anchor, 1, alt_.data(), alt_.size(), kIns);
    }

    void emit_deletion(int len, int64_t rpos1, const char *refspan, int64_t span_start1, int64_t span_len) {
        const int64_t off = rpos1 - span_start1;
        if (off < 0 || off + len > span_len) return;
        del_.assign(refspan + off, (size_t)len);
        for (char c : del_)
            if (!c || c == 'N') return;  // deleted bases unknown; cannot form a REF allele

        int64_t first1 = rpos1;
        left_shift(del_, first1);
        const int64_t anchor1 = first1 - 1;
        if (anchor1 < 1) {
            stats_.skipped_no_anchor++;
            return;
        }
        const char anchor = anchor_base(anchor1, refspan, span_start1, span_len);
        if (!anchor) {
            stats_.skipped_no_anchor++;
            return;
        }
        ref_alleles_.assign(1, anchor);
        ref_alleles_ += del_;
        add_allele(anchor1, ref_alleles_.data(), ref_alleles_.size(), &anchor, 1, kDel);
    }

    // Materialise reference bases for the read's aligned span. Returns nullptr when
    // neither the FASTA nor the MD tag can supply them.
    const char *build_refspan(bam1_t *b, int64_t start1, int64_t span_len, const uint32_t *cig, int ncig,
                              const uint8_t *qseq) {
        if (ref_.ok()) {
            const char *s = ref_.span(chrom_.c_str(), start1, span_len);
            if (s) return s;
            stats_.reads_no_ref++;
            // fall through to MD
        }
        const uint8_t *md_aux = bam_aux_get(b, "MD");
        const char *md = md_aux ? bam_aux2Z(md_aux) : nullptr;
        if (!md) {
            stats_.reads_no_ref_bases++;
            return nullptr;
        }
        stats_.reads_with_md++;

        scratch_.assign((size_t)span_len, 0);
        MDReader mdr(md);
        int64_t off = 0;
        int qpos = 0;
        for (int i = 0; i < ncig; ++i) {
            const int op = bam_cigar_op(cig[i]);
            const int len = bam_cigar_oplen(cig[i]);
            switch (op) {
                case BAM_CMATCH:
                case BAM_CEQUAL:
                case BAM_CDIFF:
                    for (int k = 0; k < len; ++k) {
                        const int rb = mdr.next_aligned();
                        if (rb < 0) {
                            stats_.reads_bad_md++;
                            return nullptr;
                        }
                        scratch_[(size_t)(off + k)] =
                            rb == 0 ? seq_nt16_str[bam_seqi(qseq, qpos + k)] : (char)rb;
                    }
                    off += len;
                    qpos += len;
                    break;
                case BAM_CDEL:
                    if (!mdr.read_deletion(len, scratch_.data() + off)) {
                        stats_.reads_bad_md++;
                        return nullptr;
                    }
                    off += len;
                    break;
                case BAM_CREF_SKIP:
                    off += len;  // left as 0: unknown, and never needed
                    break;
                case BAM_CINS:
                case BAM_CSOFT_CLIP:
                    qpos += len;
                    break;
                default:
                    break;
            }
        }
        return scratch_.data();
    }

    const Config &cfg_;
    sam_hdr_t *hdr_;
    RefCache &ref_;
    regidx_t *reg_;
    regitr_t *regitr_;
    Writer &out_;
    Stats stats_;

    std::vector<Slot> ring_;
    size_t cap_ = 0, mask_ = 0, head_ = 0;
    int64_t win_start_ = 1, win_end_ = 0;
    int64_t margin_ = 0;
    bool vaf_filter_ = false;
    int32_t tid_ = -1;
    std::string chrom_, out_chrom_;

    std::vector<char> scratch_;  // reference bases for the current read's span
    std::string ins_, del_, alt_, ref_alleles_, line_;
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------
void usage(FILE *fp) {
    fprintf(fp,
            "bam2vcf %s -- call variants straight out of BAM/CRAM alignment records\n"
            "\n"
            "Usage: bam2vcf --bam in.bam [options] > out.vcf\n"
            "\n"
            "Input/output:\n"
            "  -b, --bam FILE          coordinate-sorted BAM/CRAM (required)\n"
            "  -f, --reference FILE    reference FASTA; enables left-alignment and removes\n"
            "                          the MD-tag requirement\n"
            "  -o, --out FILE          output VCF ('-' for stdout, default); .gz/.bgz implies BGZF\n"
            "      --bgzip             force BGZF output\n"
            "      --index             tabix-index the output (BGZF only)\n"
            "  -R, --regions FILE      BED file; only variants overlapping these regions\n"
            "      --stats-json FILE   write run statistics as JSON\n"
            "\n"
            "Filters:\n"
            "  -c, --min-count N       minimum supporting reads per allele, i.e. an AO filter (default 1)\n"
            "      --min-vaf F         minimum alternate allele fraction AO/DP, inclusive (default 0)\n"
            "      --max-vaf F         maximum alternate allele fraction AO/DP, inclusive (default 1);\n"
            "                          lower it to drop likely-germline homozygous sites\n"
            "  -q, --min-mapq N        minimum read mapping quality (default 0)\n"
            "  -Q, --min-baseq N       minimum base quality at the variant (default 0)\n"
            "  -F, --exclude-flags N   skip reads with any of these flags (default 0xF04:\n"
            "                          unmapped, secondary, QC fail, duplicate, supplementary)\n"
            "  -G, --require-flags N   skip reads missing any of these flags (default 0)\n"
            "      --max-reads N       stop after N reads (0 = all)\n"
            "\n"
            "Behaviour:\n"
            "  -t, --threads N         BGZF worker threads (default 1)\n"
            "      --no-normalize      report indels as aligned instead of left-aligning them\n"
            "      --skip-indels       report substitutions only\n"
            "      --flush-margin N    left-alignment lookback in bp (default 1000)\n"
            "      --no-type           omit the TYPE INFO field\n"
            "      --strip-version     drop trailing .N version suffixes from contig names\n"
            "      --assume-sorted     skip the @HD SO:coordinate header check\n"
            "  -p, --progress          log progress to stderr\n"
            "  -h, --help              this message\n",
            BAM2VCF_VERSION);
}

bool ends_with(const std::string &s, const char *suf) {
    size_t n = strlen(suf);
    return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

bool header_is_coordinate_sorted(sam_hdr_t *hdr) {
    kstring_t val = KS_INITIALIZE;
    bool sorted = false;
    if (sam_hdr_find_tag_hd(hdr, "SO", &val) == 0 && val.s) sorted = strcmp(val.s, "coordinate") == 0;
    ks_free(&val);
    return sorted;
}

void write_stats(const Config &cfg, const Stats &st, double elapsed) {
    if (cfg.stats_json.empty()) return;
    FILE *fp = cfg.stats_json == "-" ? stderr : fopen(cfg.stats_json.c_str(), "w");
    if (!fp) return;
    fprintf(fp,
            "{\"reads_total\":%" PRId64 ",\"reads_used\":%" PRId64 ",\"reads_skipped_flag\":%" PRId64
            ",\"reads_skipped_mapq\":%" PRId64 ",\"reads_skipped_region\":%" PRId64 ",\"reads_fast_path\":%" PRId64
            ",\"reads_with_md\":%" PRId64 ",\"reads_bad_md\":%" PRId64 ",\"reads_no_ref\":%" PRId64
            ",\"reads_without_ref_bases\":%" PRId64 ",\"alleles_seen\":%" PRId64 ",\"records_emitted\":%" PRId64
            ",\"sites_emitted\":%" PRId64 ",\"dropped_min_count\":%" PRId64 ",\"dropped_vaf\":%" PRId64
            ",\"dropped_region\":%" PRId64
            ",\"skipped_no_anchor\":%" PRId64 ",\"shifts_clamped\":%" PRId64 ",\"seconds\":%.3f}\n",
            st.reads_total, st.reads_used, st.reads_skipped_flag, st.reads_skipped_mapq, st.reads_skipped_region,
            st.reads_fast_path, st.reads_with_md, st.reads_bad_md, st.reads_no_ref, st.reads_no_ref_bases,
            st.alleles_seen, st.records_emitted, st.sites_emitted, st.dropped_min_count, st.dropped_vaf,
            st.dropped_region, st.skipped_no_anchor, st.shifts_clamped, elapsed);
    if (fp != stderr) fclose(fp);
}

}  // namespace

int main(int argc, char **argv) {
    Config cfg;
    bool bgzip_set = false;

    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        auto need = [&](const char *what) -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "bam2vcf: %s requires a value\n", what);
                exit(2);
            }
            return argv[++i];
        };
        if (!strcmp(a, "-b") || !strcmp(a, "--bam"))
            cfg.bam = need(a);
        else if (!strcmp(a, "-f") || !strcmp(a, "--reference") || !strcmp(a, "--fasta"))
            cfg.reference = need(a);
        else if (!strcmp(a, "-o") || !strcmp(a, "--out") || !strcmp(a, "--output"))
            cfg.out = need(a);
        else if (!strcmp(a, "-R") || !strcmp(a, "--regions"))
            cfg.regions = need(a);
        else if (!strcmp(a, "--stats-json"))
            cfg.stats_json = need(a);
        else if (!strcmp(a, "-c") || !strcmp(a, "--min-count"))
            cfg.min_count = (uint32_t)strtoul(need(a), nullptr, 10);
        else if (!strcmp(a, "--min-vaf"))
            cfg.min_vaf = strtod(need(a), nullptr);
        else if (!strcmp(a, "--max-vaf"))
            cfg.max_vaf = strtod(need(a), nullptr);
        else if (!strcmp(a, "-q") || !strcmp(a, "--min-mapq"))
            cfg.min_mapq = (int)strtol(need(a), nullptr, 10);
        else if (!strcmp(a, "-Q") || !strcmp(a, "--min-baseq"))
            cfg.min_baseq = (int)strtol(need(a), nullptr, 10);
        else if (!strcmp(a, "-F") || !strcmp(a, "--exclude-flags"))
            cfg.exclude_flags = (uint16_t)strtoul(need(a), nullptr, 0);
        else if (!strcmp(a, "-G") || !strcmp(a, "--require-flags"))
            cfg.require_flags = (uint16_t)strtoul(need(a), nullptr, 0);
        else if (!strcmp(a, "--max-reads"))
            cfg.max_reads = strtoll(need(a), nullptr, 10);
        else if (!strcmp(a, "-t") || !strcmp(a, "--threads"))
            cfg.threads = (int)strtol(need(a), nullptr, 10);
        else if (!strcmp(a, "--flush-margin"))
            cfg.flush_margin = strtoll(need(a), nullptr, 10);
        else if (!strcmp(a, "--no-normalize"))
            cfg.normalize = false;
        else if (!strcmp(a, "--skip-indels"))
            cfg.skip_indels = true;
        else if (!strcmp(a, "--no-type"))
            cfg.emit_type = false;
        else if (!strcmp(a, "--bgzip")) {
            cfg.bgzip = true;
            bgzip_set = true;
        } else if (!strcmp(a, "--index"))
            cfg.index = true;
        else if (!strcmp(a, "--strip-version"))
            cfg.strip_version = true;
        else if (!strcmp(a, "--assume-sorted"))
            cfg.assume_sorted = true;
        else if (!strcmp(a, "-p") || !strcmp(a, "--progress"))
            cfg.progress = true;
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            usage(stdout);
            return 0;
        } else if (!strcmp(a, "--version")) {
            printf("%s\n", BAM2VCF_VERSION);
            return 0;
        } else {
            fprintf(stderr, "bam2vcf: unknown option '%s'\n", a);
            usage(stderr);
            return 2;
        }
    }

    if (cfg.bam.empty()) {
        fprintf(stderr, "bam2vcf: --bam is required\n");
        usage(stderr);
        return 2;
    }
    if (!bgzip_set && (ends_with(cfg.out, ".gz") || ends_with(cfg.out, ".bgz"))) cfg.bgzip = true;
    if (cfg.index && !cfg.bgzip) {
        fprintf(stderr, "bam2vcf: --index requires BGZF output (use a .gz output path or --bgzip)\n");
        return 2;
    }
    if (cfg.threads < 1) cfg.threads = 1;
    if (cfg.flush_margin < 0) cfg.flush_margin = 0;
    if (cfg.min_vaf < 0.0 || cfg.min_vaf > 1.0 || cfg.max_vaf < 0.0 || cfg.max_vaf > 1.0) {
        fprintf(stderr, "bam2vcf: --min-vaf and --max-vaf must lie between 0 and 1\n");
        return 2;
    }
    if (cfg.min_vaf > cfg.max_vaf) {
        fprintf(stderr, "bam2vcf: --min-vaf (%g) exceeds --max-vaf (%g), so nothing could pass\n", cfg.min_vaf,
                cfg.max_vaf);
        return 2;
    }

    samFile *in = sam_open(cfg.bam.c_str(), "r");
    if (!in) {
        fprintf(stderr, "bam2vcf: cannot open '%s'\n", cfg.bam.c_str());
        return 1;
    }

    htsThreadPool pool = {nullptr, 0};
    if (cfg.threads > 1) {
        pool.pool = hts_tpool_init(cfg.threads);
        if (pool.pool) hts_set_opt(in, HTS_OPT_THREAD_POOL, &pool);
    }
    if (!cfg.reference.empty()) hts_set_fai_filename(in, cfg.reference.c_str());  // for CRAM

    sam_hdr_t *hdr = sam_hdr_read(in);
    if (!hdr) {
        fprintf(stderr, "bam2vcf: cannot read header of '%s'\n", cfg.bam.c_str());
        return 1;
    }
    if (!cfg.assume_sorted && !header_is_coordinate_sorted(hdr)) {
        fprintf(stderr,
                "bam2vcf: '%s' is not marked coordinate sorted (@HD SO:coordinate).\n"
                "  Depth accounting and sorted VCF output require coordinate order.\n"
                "  Run `samtools sort -o sorted.bam %s`, or pass --assume-sorted if the\n"
                "  header is merely missing the tag.\n",
                cfg.bam.c_str(), cfg.bam.c_str());
        return 1;
    }

    faidx_t *fai = nullptr;
    if (!cfg.reference.empty()) {
        fai = fai_load(cfg.reference.c_str());
        if (!fai) {
            fprintf(stderr, "bam2vcf: cannot load reference FASTA index for '%s'\n", cfg.reference.c_str());
            return 1;
        }
    }
    RefCache refcache(fai);

    regidx_t *reg = nullptr;
    regitr_t *regitr = nullptr;
    if (!cfg.regions.empty()) {
        reg = regidx_init(cfg.regions.c_str(), regidx_parse_bed, nullptr, 0, nullptr);
        if (!reg) {
            fprintf(stderr, "bam2vcf: cannot read regions BED '%s'\n", cfg.regions.c_str());
            return 1;
        }
        regitr = regitr_init(reg);
    }

    Writer out;
    if (!out.open(cfg.out, cfg.bgzip, &pool)) {
        fprintf(stderr, "bam2vcf: cannot open output '%s'\n", cfg.out.c_str());
        return 1;
    }

    Caller caller(cfg, hdr, refcache, reg, regitr, out);
    caller.write_header();
    Stats &st = caller.stats();

    bam1_t *b = bam_init1();
    clock_t t0 = clock();
    int ret;
    while ((ret = sam_read1(in, hdr, b)) >= 0) {
        st.reads_total++;
        if (cfg.progress && (st.reads_total % 5000000) == 0)
            fprintf(stderr, "bam2vcf: %" PRId64 "M reads, %" PRId64 " records written\n", st.reads_total / 1000000,
                    st.records_emitted);
        if (cfg.max_reads && st.reads_total > cfg.max_reads) {
            st.reads_total--;
            break;
        }
        const uint16_t flag = b->core.flag;
        if (flag & cfg.exclude_flags) {
            st.reads_skipped_flag++;
            continue;
        }
        if (cfg.require_flags && (flag & cfg.require_flags) != cfg.require_flags) {
            st.reads_skipped_flag++;
            continue;
        }
        if (b->core.tid < 0 || b->core.n_cigar == 0) {
            st.reads_skipped_flag++;
            continue;
        }
        if (b->core.qual < cfg.min_mapq) {
            st.reads_skipped_mapq++;
            continue;
        }
        if (reg) {
            const char *chrom = sam_hdr_tid2name(hdr, b->core.tid);
            int64_t beg0 = b->core.pos;
            int64_t end0 = bam_endpos(b) - 1;
            if (end0 < beg0) end0 = beg0;
            if (!chrom || !regidx_overlap(reg, chrom, beg0, end0, regitr)) {
                st.reads_skipped_region++;
                continue;
            }
        }
        st.reads_used++;
        caller.add_read(b);
    }
    caller.finish();
    const double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;

    bool ok = out.close();
    if (ret < -1) {
        fprintf(stderr, "bam2vcf: truncated or corrupt input after %" PRId64 " reads\n", st.reads_total);
        ok = false;
    }

    // Without an MD tag or a FASTA there is nothing to read mismatches and deletions out
    // of, and the walk would quietly return an insertion-only result rather than failing.
    if (st.reads_used > 0 && st.reads_no_ref_bases == st.reads_used - st.reads_fast_path && st.reads_with_md == 0 &&
        !fai) {
        fprintf(stderr,
                "bam2vcf: no read in '%s' carries an MD tag, so mismatches and deletions cannot be\n"
                "  identified. Pass --reference <ref.fa>, re-align emitting MD (for STAR:\n"
                "  --outSAMattributes NH HI AS nM MD), or add it with\n"
                "  `samtools calmd -b %s ref.fa > with_md.bam`.\n",
                cfg.bam.c_str(), cfg.bam.c_str());
        ok = false;
    } else if (st.reads_no_ref_bases > 0) {
        fprintf(stderr, "bam2vcf: warning: %" PRId64 " of %" PRId64 " reads lacked usable reference bases and were skipped\n",
                st.reads_no_ref_bases, st.reads_used);
    }
    if (st.reads_bad_md > 0)
        fprintf(stderr, "bam2vcf: warning: %" PRId64 " reads had a malformed MD tag and were skipped\n", st.reads_bad_md);
    if (st.reads_no_ref > 0)
        fprintf(stderr, "bam2vcf: warning: %" PRId64 " reads mapped to contigs absent from the reference FASTA\n",
                st.reads_no_ref);

    if (ok && cfg.index && cfg.out != "-") {
        if (tbx_index_build(cfg.out.c_str(), 0, &tbx_conf_vcf) < 0) {
            fprintf(stderr, "bam2vcf: warning: could not tabix-index '%s'\n", cfg.out.c_str());
        }
    }

    write_stats(cfg, st, elapsed);
    if (cfg.progress)
        fprintf(stderr, "bam2vcf: %" PRId64 " reads, %" PRId64 " records, %.1fs\n", st.reads_total, st.records_emitted,
                elapsed);

    bam_destroy1(b);
    sam_hdr_destroy(hdr);
    sam_close(in);
    if (fai) fai_destroy(fai);
    if (regitr) regitr_destroy(regitr);
    if (reg) regidx_destroy(reg);
    if (pool.pool) hts_tpool_destroy(pool.pool);
    return ok ? 0 : 1;
}
