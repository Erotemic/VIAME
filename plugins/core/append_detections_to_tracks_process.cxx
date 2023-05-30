/**
 * \file
 * \brief Append a detected object set to an object track set
 */

#include "append_detections_to_tracks_process.h"

#include <vital/vital_types.h>
#include <vital/types/image_container.h>
#include <vital/types/timestamp.h>
#include <vital/types/timestamp_config.h>
#include <vital/types/object_track_set.h>

#include <sprokit/processes/kwiver_type_traits.h>
#include <sprokit/pipeline/process_exception.h>

#include <memory>


namespace kv = kwiver::vital;

namespace viame
{

namespace core
{

create_config_trait( min_frame_count, unsigned, "0",
  "If set, generate an appended detected object to an object track set for frames after min_frame_count" );
create_config_trait( max_frame_count, unsigned, "0",
  "If set, generate an appended detected object to an object track set for frames before max_frame_count" );
create_config_trait( do_wait_process_end_before_sending_output, bool, "0",
  "If set, waits until the in port detection set is at the end before sending the results" );

// =============================================================================
// Private implementation class
class append_detections_to_tracks_process::priv
{
public:
  explicit priv( append_detections_to_tracks_process* parent );
  ~priv();

  // Configuration settings
  unsigned m_min_frame_count;
  unsigned m_max_frame_count;
  bool m_do_wait_process_end_before_sending_output;
  unsigned m_max_detection{};

  // Internal variables
  unsigned m_track_counter;
  unsigned m_frame_counter;
  kv::object_track_set_sptr m_output{};
  std::vector<std::vector< kv::track_state_sptr >> m_states;

  // Other variables
  append_detections_to_tracks_process* parent;

  kv::logger_handle_t m_logger;
};


// -----------------------------------------------------------------------------
append_detections_to_tracks_process::priv
::priv( append_detections_to_tracks_process* ptr )
  : m_min_frame_count( 0 )
  , m_max_frame_count( 0 )
  , m_do_wait_process_end_before_sending_output( false )
  , m_track_counter( 0 )
  , m_frame_counter( 0 )
  , parent( ptr )
  , m_logger( kv::get_logger( "append_detections_to_tracks_process" ) )
{
}


append_detections_to_tracks_process::priv
::~priv()
{
}


// =============================================================================
append_detections_to_tracks_process
::append_detections_to_tracks_process( kv::config_block_sptr const& config )
  : process( config ),
    d( new append_detections_to_tracks_process::priv( this ) )
{
  make_ports();
  make_config();
}


append_detections_to_tracks_process
::~append_detections_to_tracks_process()
{
}


// -----------------------------------------------------------------------------
void
append_detections_to_tracks_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;

  required.insert( flag_required );

  // -- inputs --
  declare_input_port_using_trait( timestamp, required );
  declare_input_port_using_trait( detected_object_set, required );

  // -- outputs --
  declare_output_port_using_trait( timestamp, optional );
  declare_output_port_using_trait( object_track_set, optional );
}


// -----------------------------------------------------------------------------
void
append_detections_to_tracks_process
::make_config()
{
  declare_config_using_trait( min_frame_count );
  declare_config_using_trait( max_frame_count );
  declare_config_using_trait( do_wait_process_end_before_sending_output );
}


// -----------------------------------------------------------------------------
void
append_detections_to_tracks_process
::_configure()
{
  d->m_min_frame_count = config_value_using_trait( min_frame_count );
  d->m_max_frame_count = config_value_using_trait( max_frame_count );
  d->m_do_wait_process_end_before_sending_output = config_value_using_trait( do_wait_process_end_before_sending_output );

  if ( d->m_min_frame_count > d->m_max_frame_count )
  {
    std::stringstream ss;
    ss  << "Invalid min/max frame count limits (" << d->m_min_frame_count << ", " << d->m_max_frame_count << ")";
    VITAL_THROW( sprokit::invalid_configuration_exception, name(), ss.str());
  }
}

// -----------------------------------------------------------------------------
void
append_detections_to_tracks_process
::_step()
{
  kv::image_container_sptr image;
  kv::timestamp timestamp;
  kv::detected_object_set_sptr detections;

  timestamp = grab_from_port_using_trait( timestamp );
  detections = grab_from_port_using_trait( detected_object_set );

  d->m_max_detection = std::max((int) d->m_max_detection,(int) detections->size());

  if( !d->m_max_frame_count ||
      (timestamp.get_frame() >= d->m_min_frame_count && timestamp.get_frame() <= d->m_max_frame_count))
  {
    // init track states if m_track_counter == 0
    if(d->m_max_detection > d->m_states.size())
    {
      d->m_states.resize(d->m_max_detection);
    }

    if( !d->m_states.empty() &&  d->m_states.size() == detections->size())
    {
      std::vector< kv::track_sptr > all_tracks;
      for( unsigned detectId = 0; detectId < detections->size(); ++detectId )
      {
        d->m_states[detectId].push_back(
              std::make_shared< kv::object_track_state >(
                timestamp, detections->at( detectId ) ) );

        kv::track_sptr ot = kv::track::create();
        ot->set_id( detectId );

        for( const auto& state : d->m_states[detectId] )
        {
          ot->append( state );
        }

        all_tracks.push_back(ot);
      }

      d->m_output = std::make_shared<kv::object_track_set>(all_tracks);
      d->m_track_counter++;
    }
  }
  d->m_frame_counter++;
  LOG_DEBUG(d->m_logger, "Accumulated non empty tracks (" << d->m_track_counter << "/" << d->m_frame_counter << ")");

  // Send the object tracks through the output port in case the "wait_process_end" flag is set to false (default) or
  // if the process has reached its end.
  // Otherwise, send an empty datum in the output ports
  auto port_info = peek_at_port_using_trait(detected_object_set);
  auto is_input_complete = port_info.datum->type() == sprokit::datum::complete;
  if (!d->m_do_wait_process_end_before_sending_output || is_input_complete) {
    LOG_DEBUG(d->m_logger, "Sending appended object tracks.");
    push_to_port_using_trait(timestamp, timestamp);
    push_to_port_using_trait(object_track_set, d->m_output);
  } else {
    LOG_DEBUG(d->m_logger, "Sending empty.");
    const auto dat = sprokit::datum::empty_datum();
    push_datum_to_port_using_trait(timestamp, dat);
    push_datum_to_port_using_trait(object_track_set, dat);
  }

  if (is_input_complete) {
    mark_process_as_complete();
  }
}

} // end namespace core

} // end namespace viame
