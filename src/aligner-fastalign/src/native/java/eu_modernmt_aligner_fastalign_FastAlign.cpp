//
// Created by Davide  Caroselli on 26/08/16.
//

#include "javah/eu_modernmt_aligner_fastalign_FastAlign.h"
#include "fastalign/FastAligner.h"
#include "jniutil.h"

#ifdef _OPENMP
#include <omp.h>
#endif


using namespace std;
using namespace mmt;
using namespace mmt::fastalign;

inline void ParseSentence(JNIEnv *jvm, jobjectArray jarray, vector<string> &output) {
    jsize size = jvm->GetArrayLength(jarray);
    output.reserve((size_t) size);

    for (jsize i = 0; i < size; i++) {
        jstring jword = (jstring) jvm->GetObjectArrayElement(jarray, i);
        string word = jni_jstrtostr(jvm, jword);

        output.push_back(word);
    }
}

inline jintArray AlignmentToArray(JNIEnv *jvm, alignment_t align, bool reversed) {
    jsize hsize = (jsize) align.points.size();
    jsize size = (jsize) (hsize * 2);

    jint *buffer = new jint[size];
    for (jsize i = 0; i < hsize; i++) {
        pair<length_t, length_t> &pair = align.points[i];
        buffer[i] = reversed ? pair.second : pair.first;
        buffer[i + hsize] = reversed ? pair.first : pair.second;
    }

    jintArray jarray = jvm->NewIntArray(size);
    jvm->SetIntArrayRegion(jarray, 0, size, buffer);
    delete[] buffer;

    return jarray;
}

/*
 * Class:     eu_modernmt_aligner_fastalign_FastAlign
 * Method:    instantiate
 * Signature: (Ljava/lang/String;I)J
 */
JNIEXPORT jlong JNICALL
Java_eu_modernmt_aligner_fastalign_FastAlign_instantiate(JNIEnv *jvm, jobject jself, jstring jmodel, jint threads) {
    string modelPath = jni_jstrtostr(jvm, jmodel);
    return (jlong) new FastAligner(modelPath, threads);
}

/*
 * Class:     eu_modernmt_aligner_fastalign_FastAlign
 * Method:    align
 * Signature: (JZ[Ljava/lang/String;[Ljava/lang/String;I[[I)F
 */
JNIEXPORT jfloat JNICALL
Java_eu_modernmt_aligner_fastalign_FastAlign_align__JZ_3Ljava_lang_String_2_3Ljava_lang_String_2I_3_3I
        (JNIEnv *jvm, jobject jself, jlong jhandle, jboolean reversed,
         jobjectArray jsource, jobjectArray jtarget, jint jstrategy, jobjectArray joutput) {
    FastAligner *aligner = reinterpret_cast<FastAligner *>(jhandle);

    vector<string> source, target;
    ParseSentence(jvm, reversed ? jtarget : jsource, source);
    ParseSentence(jvm, reversed ? jsource : jtarget, target);

    alignment_t align = aligner->GetAlignment(source, target, (Symmetrization) jstrategy);
    jintArray alignment = AlignmentToArray(jvm, align, (bool) reversed);
    jvm->SetObjectArrayElement(joutput, 0, alignment);

    return (jfloat) align.score;
}

/*
 * Class:     eu_modernmt_aligner_fastalign_FastAlign
 * Method:    align
 * Signature: (JZ[[Ljava/lang/String;[[Ljava/lang/String;I[[I)[F
 */
JNIEXPORT jfloatArray JNICALL
Java_eu_modernmt_aligner_fastalign_FastAlign_align__JZ_3_3Ljava_lang_String_2_3_3Ljava_lang_String_2I_3_3I
        (JNIEnv *jvm, jobject jself, jlong jhandle, jboolean reversed,
         jobjectArray jsources, jobjectArray jtargets, jint jstrategy, jobjectArray joutput) {
    FastAligner *aligner = reinterpret_cast<FastAligner *>(jhandle);
    jsize length = jvm->GetArrayLength(jsources);

    vector<pair<vector<string>, vector<string>>> batch;
    batch.reserve((size_t) length);

    for (jsize i = 0; i < length; i++) {
        jobjectArray jsource = (jobjectArray) jvm->GetObjectArrayElement(jsources, i);
        jobjectArray jtarget = (jobjectArray) jvm->GetObjectArrayElement(jtargets, i);

        vector<string> source, target;
        ParseSentence(jvm, reversed ? jtarget : jsource, source);
        ParseSentence(jvm, reversed ? jsource : jtarget, target);

        batch.push_back(pair<vector<string>, vector<string>>(source, target));
    }

    vector<alignment_t> alignments;
    aligner->GetAlignments(batch, alignments, (Symmetrization) jstrategy);

    jsize size = (jsize) alignments.size();

    jfloat *jscores = new jfloat[size];
    for (jsize i = 0; i < size; i++) {
        jintArray alignment = AlignmentToArray(jvm, alignments[i], (bool) reversed);
        jvm->SetObjectArrayElement(joutput, i, alignment);
        jscores[i] = (jfloat) alignments[i].score;
    }

    jfloatArray jarray = jvm->NewFloatArray(size);
    jvm->SetFloatArrayRegion(jarray, 0, size, jscores);
    delete[] jscores;

    return jarray;
}

/*
 * Class:     eu_modernmt_aligner_fastalign_FastAlign
 * Method:    dispose
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_eu_modernmt_aligner_fastalign_FastAlign_dispose(JNIEnv *jvm, jobject jself, jlong ptr) {
    if (ptr != 0) {
        FastAligner *instance = (FastAligner *) ptr;
        delete instance;
    }

    return 0;
}