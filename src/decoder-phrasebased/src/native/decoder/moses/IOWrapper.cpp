// $Id$

/***********************************************************************
Moses - factored phrase-based language decoder
Copyright (c) 2006 University of Edinburgh
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
			this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
			this list of conditions and the following disclaimer in the documentation
			and/or other materials provided with the distribution.
 * Neither the name of the University of Edinburgh nor the names of its contributors
			may be used to endorse or promote products derived from this software
			without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
 ***********************************************************************/

#include <iostream>
#include <stack>
#include <boost/algorithm/string.hpp>
#include <boost/foreach.hpp>

#include "Hypothesis.h"
#include "StaticData.h"
#include "InputFileStream.h"
#include "FF/StatefulFeatureFunction.h"

#include "IOWrapper.h"

#include <boost/filesystem.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filter/bzip2.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>

using namespace std;

namespace Moses
{

IOWrapper::IOWrapper(AllOptions const& opts)
  : m_options(new AllOptions(opts))
  , m_nBestStream(NULL)
  , m_surpressSingleBestOutput(false)
  , m_look_ahead(0)
  , m_look_back(0)
  , m_buffered_ahead(0)
  , spe_src(NULL)
  , spe_trg(NULL)
  , spe_aln(NULL)
{
  const StaticData &staticData = StaticData::Instance();
  Parameter const& P = staticData.GetParameter();

  // context buffering for context-sensitive decoding
  m_look_ahead = m_options->context.look_ahead;
  m_look_back  = m_options->context.look_back;
  m_inputType  = m_options->input.input_type;

  UTIL_THROW_IF2((m_look_ahead || m_look_back) && m_inputType != SentenceInput,
                 "Context-sensitive decoding currently works only with sentence input.");

  m_currentLine = m_options->output.start_translation_id;
  m_inputFactorOrder = &m_options->input.factor_order;

  size_t nBestSize = m_options->nbest.nbest_size;
  string nBestFilePath = m_options->nbest.output_file_path;

  staticData.GetParameter().SetParameter<string>(m_inputFilePath, "input-file", "");
  if (m_inputFilePath.empty()) {
    m_inputFile = NULL;
    m_inputStream = &cin;
  } else {
    VERBOSE(2,"IO from File" << endl);
    m_inputFile = new InputFileStream(m_inputFilePath);
    m_inputStream = m_inputFile;
  }

  if (nBestSize > 0) {
    m_nBestOutputCollector.reset(new Moses::OutputCollector(nBestFilePath));
    if (m_nBestOutputCollector->OutputIsCout()) {
      m_surpressSingleBestOutput = true;
    }
  }

  std::string path;
  P.SetParameter<std::string>(path, "output-search-graph-extended", "");
  if (!path.size()) P.SetParameter<std::string>(path, "output-search-graph", "");
  if (path.size()) m_searchGraphOutputCollector.reset(new OutputCollector(path));

  P.SetParameter<std::string>(path, "output-unknowns", "");
  if (path.size()) m_unknownsCollector.reset(new OutputCollector(path));

  if (!m_surpressSingleBestOutput) {
    m_singleBestOutputCollector.reset(new Moses::OutputCollector(&std::cout));
  }

}

IOWrapper::~IOWrapper()
{
  if (m_inputFile != NULL)
    delete m_inputFile;
  // if (m_nBestStream != NULL && !m_surpressSingleBestOutput) {
  // outputting n-best to file, rather than stdout. need to close file and delete obj
  // delete m_nBestStream;
  // }

  // delete m_detailedTranslationReportingStream;
  // delete m_alignmentInfoStream;
  // delete m_unknownsStream;
  // delete m_outputSearchGraphStream;
  // delete m_outputWordGraphStream;
  // delete m_latticeSamplesStream;
}

// InputType*
// IOWrapper::
// GetInput(InputType* inputType)
// {
//   if(inputType->Read(*m_inputStream, *m_inputFactorOrder)) {
//     return inputType;
//   } else {
//     delete inputType;
//     return NULL;
//   }
// }

boost::shared_ptr<InputType>
IOWrapper::
GetBufferedInput()
{
  switch(m_inputType) {
  case SentenceInput:
    return BufferInput<Sentence>();
  default:
    TRACE_ERR("Unknown input type: " << m_inputType << "\n");
    return boost::shared_ptr<InputType>();
  }

}

boost::shared_ptr<InputType>
IOWrapper::
ReadInput(boost::shared_ptr<std::vector<std::string> >* cw)
{
#ifdef WITH_THREADS
  boost::lock_guard<boost::mutex> lock(m_lock);
#endif
  boost::shared_ptr<InputType> source = GetBufferedInput();
  if (source) {
    source->SetTranslationId(m_currentLine++);

    // when using a sliding context window, remove obsolete past input from buffer:
    if (m_past_input.size() && m_look_back != std::numeric_limits<size_t>::max()) {
      list<boost::shared_ptr<InputType> >::iterator m = m_past_input.end();
      for (size_t cnt = 0; cnt < m_look_back && --m != m_past_input.begin();)
        cnt += (*m)->GetSize();
      while (m_past_input.begin() != m) m_past_input.pop_front();
    }

    if (m_look_back)
      m_past_input.push_back(source);
  }
  if (cw) *cw = GetCurrentContextWindow();
  return source;
}

boost::shared_ptr<std::vector<std::string> >
IOWrapper::
GetCurrentContextWindow() const
{
  boost::shared_ptr<std::vector<string> > context(new std::vector<string>);
  BOOST_FOREACH(boost::shared_ptr<InputType> const& i, m_past_input)
  context->push_back(i->ToString());
  BOOST_FOREACH(boost::shared_ptr<InputType> const& i, m_future_input)
  context->push_back(i->ToString());
  return context;
}



std::string
IOWrapper::
GetHypergraphOutputFileName(size_t const id) const
{
  return str(boost::format(m_hypergraph_output_filepattern) % id);
}


} // namespace

